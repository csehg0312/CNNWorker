module RedisQueueWatcher

# using ..CNNWorker
using ..SharedState
using ..ODESolver
using Distributed, SharedArrays, DistributedArrays
using Redis, JSON, WebSockets, CUDA
include("SocketLogger.jl")
export watch_redis_queue

function manage_workers(queue_name, socket_conn, redis_client, interrupt_flag::SharedState.InterruptFlag)
    last_worker_count = nworkers()
    
    while !interrupt_flag.value
        try
            num_tasks = Redis.llen(redis_client, queue_name)
            current_workers = nworkers()
            
            if current_workers != last_worker_count || num_tasks > 0
                SocketLogger.write_log_to_socket(socket_conn, 
                    "Queue status: $num_tasks tasks, $current_workers workers\n")
                last_worker_count = current_workers
            end
            
            if num_tasks > 10 && current_workers < min(10, Sys.CPU_THREADS - 1)
                new_workers = min(2, min(10, Sys.CPU_THREADS - 1) - current_workers)
                if new_workers > 0
                    addprocs(new_workers)
                    @everywhere workers()[end-new_workers+1:end] begin
                        using SharedArrays, FFTW, LoopVectorization, DifferentialEquations
                        using Images, FileIO, Base64, WebSockets, DistributedArrays, LinearAlgebra
                        include("cnn_worker_functions.jl")
                    end
                    SocketLogger.write_log_to_socket(socket_conn, 
                        "Added $new_workers workers. Total workers: $(nworkers())\n")
                end
            elseif num_tasks < 3 && current_workers > 2
                workers_to_remove = min(2, current_workers - 2)
                if workers_to_remove > 0
                    worker_ids = workers()[end-workers_to_remove+1:end]
                    rmprocs(worker_ids)
                    SocketLogger.write_log_to_socket(socket_conn, 
                        "Removed $workers_to_remove workers. Total workers: $(nworkers())\n")
                end
            end
        catch e
            SocketLogger.write_log_to_socket(socket_conn, 
                "Error in worker management: $e\n")
        end
        sleep(5)
    end
end

function watch_redis_queue(redis_client, queue_name, socket_conn, interrupt_flag::SharedState.InterruptFlag)
    SocketLogger.write_log_to_socket(socket_conn, "Started Redis Queue Watcher!\n")
    
    if redis_client === nothing || !is_redis_connected(redis_client)
        SocketLogger.write_log_to_socket(socket_conn, "Creating new Redis connection...\n")
        redis_client = Redis.connect(redis_client.host, redis_client.port)
    end
    
    worker_manager = @async manage_workers(queue_name, socket_conn, redis_client, interrupt_flag)

    try
        while !interrupt_flag.value
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
            catch e
                if e isa InterruptException
                    SocketLogger.write_log_to_socket(socket_conn, 
                        "Received interrupt signal in queue watcher. Shutting down gracefully...\n")
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
        try
            if @isdefined(worker_manager) && !istaskdone(worker_manager)
                schedule(worker_manager, InterruptException(), error=true)
            end
        catch
        end
        
        cleanup(redis_client, socket_conn)
        SocketLogger.write_log_to_socket(socket_conn, "Redis queue watcher shutting down...\n")
    end
end

end # module