module RedisQueueWatcher

# using ..CNNWorker

using ..ODESolver
using Distributed, SharedArrays, DistributedArrays
using Redis, JSON, WebSockets, CUDA
include("SocketLogger.jl")
export watch_redis_queue

# Create a file-based interrupt detection mechanism
const INTERRUPT_FLAG_FILE = joinpath(tempdir(), "julia_interrupt_flag.txt")

# Check if interrupt file exists
function is_interrupted()
    return isfile(INTERRUPT_FLAG_FILE)
end

# Set the interrupt flag by creating or removing the file
function set_interrupt_flag(value::Bool)
    if value
        # Create the flag file if it doesn't exist
        open(INTERRUPT_FLAG_FILE, "w") do io
            write(io, string(time()))  # Just write the current timestamp
        end
    else
        # Remove the flag file if it exists
        isfile(INTERRUPT_FLAG_FILE) && rm(INTERRUPT_FLAG_FILE)
    end
    return value
end

# Clear the flag file on module load
function __init__()
    set_interrupt_flag(false)
end

export is_interrupted, set_interrupt_flag

function is_redis_connected(redis_client)
    try
        Redis.ping(redis_client)
        return true
    catch
        return false
    end
end

function create_redis_connection(conn,host="localhost", port=6379, retries=5, delay=1)
    for attempt in 1:retries
        try
            answer = Redis.ping(Redis.RedisConnection(host=host, port=port))
            if answer == "PONG"
                SocketLogger.write_log_to_socket(conn, "Connected to at $host:$port\n")
                return Redis.RedisConnection(host=host, port=port)
            end
        catch e 
            error("Failed to connect to Redis $e\n")
        end
    end
    SocketLogger.write_log_to_socket("Failed to connect after $retries attempts!\n")
end

function safe_close_redis(redis_client)
    try
        if redis_client !== nothing
            # Access the underlying TCP socket and close it
            if hasfield(typeof(redis_client), :socket)
                isopen(redis_client.socket) && close(redis_client.socket)
            elseif hasfield(typeof(redis_client), :transport) && 
                   hasfield(typeof(redis_client.transport), :socket)
                isopen(redis_client.transport.socket) && close(redis_client.transport.socket)
            end
            # Attempt to disconnect Redis client
            try
                Redis.disconnect(redis_client)
            catch
                # Ignore disconnect errors
            end
        end
    catch e
        return false
    end
    return true
end


function cleanup(redis_client, socket_conn)
    SocketLogger.write_log_to_socket(socket_conn, "Cleaning up resources...\n")
    try
        if redis_client !== nothing
            safe_close_redis(redis_client)
        end
    catch e
        SocketLogger.write_log_to_socket(socket_conn, "Error during Redis cleanup: $e\n")
    end
end

function watch_redis_queue(redis_client, queue_name, socket_conn)
    SocketLogger.write_log_to_socket(socket_conn, "Started Redis Queue Watcher!\n")
    
    if redis_client === nothing || !is_redis_connected(redis_client)
        SocketLogger.write_log_to_socket(socket_conn, "Creating new Redis connection...\n")
        redis_client = create_redis_connection(socket_conn)
        if redis_client !== nothing
            SocketLogger.write_log_to_socket(socket_conn, "Watching queue: $queue_name\n")
        end
    end
    
    try
        while !is_interrupted()
            
            try
                if !is_redis_connected(redis_client)
                    SocketLogger.write_log_to_socket(socket_conn, "Redis connection lost. Attempting to reconnect...\n")
                    redis_client = Redis.connect(redis_client.host, redis_client.port)
                end
                result = Redis.blpop(redis_client, queue_name, 1)
                
                if result !== nothing
                    _, task_id = result
                    stored_data = Redis.get(redis_client, "task:data:$task_id")
                    
                    if stored_data !== nothing
                        processed_data = JSON.parse(stored_data)
                        image = [Float64.(row) for row in processed_data["image"]]
                        controlB = [Float64.(row) for row in processed_data["controlB"]]
                        feedbackA = [Float64.(row) for row in processed_data["feedbackA"]]
                        t_span = convert(Vector{Float64}, processed_data["t_span"])
                        Ib = Float64(processed_data["Ib"])
                        initialCondition = Float64(processed_data["initialCondition"])

                        image_matrix = hcat(image...)
                        controlB_matrix = hcat(controlB...)
                        feedbackA_matrix = hcat(feedbackA...)

                        websocket_url = processed_data["websocket"]
                        SocketLogger.write_log_to_socket(socket_conn, "Attempting to connect to WebSocket: $websocket_url\n")

                        WebSockets.open(websocket_url) do ws
                            SocketLogger.write_log_to_socket(socket_conn, "Connected to client socket! \n")
                            WebSockets.write(ws, "Connection established")

                            ODESolver.solve_ode(socket_conn, image_matrix, Ib, 
                                                  feedbackA_matrix, controlB_matrix, 
                                                  t_span, initialCondition, ws)
                        end
                    end
                end
                # Check for interrupt again
                if is_interrupted()
                    SocketLogger.write_log_to_socket(socket_conn, "Interrupt detected after processing, breaking loop...\n")
                    break
                end
            catch e
                if e isa InterruptException
                    SocketLogger.write_log_to_socket(socket_conn, 
                        "Received interrupt signal in queue watcher. Shutting down gracefully...\n")
                    set_interrupt_flag(true)  # Propagate the interrupt
                    break
                else
                    SocketLogger.write_log_to_socket(socket_conn, "An error occurred in queue watcher: $e\n")
                    if e isa Redis.ConnectionException || e isa Base.IOError
                        sleep(1)
                    end
                end
            end
        end
    catch e
        if !(e isa InterruptException)
            SocketLogger.write_log_to_socket(socket_conn, "Unhandled error in queue watcher: $e\n")
            SocketLogger.write_log_to_socket(socket_conn, "Backtrace: $(catch_backtrace())\n")
        end
    finally
        if is_interrupted() && !global_interrupt_flag_shared[1]
            set_interrupt_flag(true)
        end
        cleanup(redis_client, socket_conn)
        SocketLogger.write_log_to_socket(socket_conn, "Redis queue watcher shutting down...\n")
    end
end

end # module