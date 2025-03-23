module CNNWorker

using FFTW, LoopVectorization, Sockets, Redis, DotEnv
using SharedArrays, Distributed, DistributedArrays, LinearAlgebra
using Images, FileIO, Base64, WebSockets, DifferentialEquations

# Include dependencies
# include("SharedState.jl")
# using .SharedState
# include("SharedLogic.jl")
# using .SharedLogic

include("SocketLogger.jl")
include("ODESolver.jl")
include("RedisQueueWatcher.jl")

include("cnn_worker_functions.jl")

# # Global variable to track interrupt state
# const global_interrupt_flag = Ref(false)

# Load environment variables
function load_environment()
    env_path = joinpath(dirname(@__DIR__), ".env")
    DotEnv.load!(env_path)
    return (
        host = ENV["JULIA_PYTHON_HOST"],
        port = parse(Int, ENV["JULIA_PYTHON_PORT3"]),
        redis_host = ENV["REDIS_HOST"],
        redis_port = parse(Int, ENV["REDIS_PORT"])
    )
end

function handle_shutdown(redis_client, socket_conn)
    try
        isopen(redis_client) && close(redis_client)
    catch e
        SocketLogger.write_log_to_socket(socket_conn, "Error during Redis cleanup: $e\n")
    end
end

function start_worker(socket_conn, redis_host, redis_port)
    redis_client = nothing
    try
        redis_client = Redis.connect(redis_host, redis_port)
        Base.sigatomic_begin()
        RedisQueueWatcher.watch_redis_queue(redis_client, "queue:task_queue", socket_conn)
        Base.sigatomic_end()
    catch e
        if e isa InterruptException
            SocketLogger.write_log_to_socket("Received interrupt signal!")
        end
        SocketLogger.write_log_to_socket(socket_conn, "Error: $e\nBacktrace: $(catch_backtrace())\n")
        rethrow(e)
    finally
        handle_shutdown(redis_client, socket_conn)
    end
end

# In CNNWorker.jl

function setup_signal_handlers()
    return @async begin
        while !global_interrupt_flag[]
            try
                sleep(0.5)  # Check more frequently
                
                # Check if the interrupt flag file exists
                if RedisQueueWatcher.is_interrupted()
                    global_interrupt_flag[] = true
                    println("\nInterrupt flag detected, shutting down...")
                    break
                end
            catch e
                if e isa InterruptException
                    println("\nReceived interrupt signal in main process")
                    global_interrupt_flag[] = true
                    # Set the shared interrupt flag file
                    RedisQueueWatcher.set_interrupt_flag(true)
                    break
                end
            end
        end
    end
end

function main()
    signal_task = nothing
    try
        signal_task = setup_signal_handlers()
        config = load_environment()
        println(typeof(config.host))
        println(typeof(config.port))
        if (conn = SocketLogger.connect_to_python_socket(config.host, config.port)) !== nothing
            SocketLogger.write_log_to_socket(conn, "Julia Worker started!\n")
            start_worker(conn, config.redis_host, config.redis_port)
        else
            println("Failed to establish connection")
        end
    catch e
        println("An error occurred: $e")
        RedisQueueWatcher.set_interrupt_flag(true)
        rethrow(e)
    finally
        RedisQueueWatcher.set_interrupt_flag(true)
        signal_task !== nothing && wait(signal_task)
        println("Julia Worker shutting down...\n")
    end
end

# Option 1: Use __init__ to run code when module is loaded
function __init__()
    # If you want to automatically run main when the module loads
    # Uncomment the next line. Otherwise, keep it commented.
    main()
end

# Option 2: Export main so it can be called after importing
export main

end # module