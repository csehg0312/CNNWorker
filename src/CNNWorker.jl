module CNNWorker

using FFTW, LoopVectorization, Sockets, Redis, DotEnv
using SharedArrays, Distributed, DistributedArrays, LinearAlgebra
using Images, FileIO, Base64, WebSockets, DifferentialEquations


# Include dependencies
include("SharedState.jl")
using .SharedState

include("SocketLogger.jl")
include("ODESolver.jl")
include("RedisQueueWatcher.jl")

include("cnn_worker_function.jl")

# Global variable to track interrupt state
const interrupt_flag = SharedState.InterruptFlag()

# Load environment variables
function load_environment()
    env_path = joinpath(dirname(@__DIR__), ".env")
    DotEnv.load!(env_path)
    return (
        host = get(ENV, "JULIA_PYTHON_HOST", "localhost"),
        port = parse(Int, get(ENV, "JULIA_PYTHON_PORT3", "5555")),
        redis_host = get(ENV, "REDIS_HOST", "localhost"),
        redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))
    )
end

function setup_distributed_environment(num_workers=Sys.CPU_THREADS - 1)
    if nprocs() == 1
        addprocs(num_workers)
        @everywhere using SharedArrays, FFTW, LoopVectorization, DifferentialEquations
        @everywhere using Images, FileIO, Base64, WebSockets, DistributedArrays, LinearAlgebra
        @everywhere include("cnn_worker_functions.jl")
    end
    return (nprocs(), nworkers(), Threads.nthreads())
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
        total_procs, worker_procs, threads_per_proc = setup_distributed_environment()
        SocketLogger.write_log_to_socket(socket_conn, "Distributed environment: $total_procs processes, $worker_procs workers, $threads_per_proc threads\n")

        redis_client = Redis.connect(redis_host, redis_port)
        RedisQueueWatcher.watch_redis_queue(redis_client, "queue:task_queue", socket_conn, interrupt_flag)
    catch e
        SocketLogger.write_log_to_socket(socket_conn, "Error: $e\nBacktrace: $(catch_backtrace())\n")
        rethrow(e)
    finally
        nworkers() > 0 && rmprocs(workers())
        handle_shutdown(redis_client, socket_conn)
    end
end

function setup_signal_handlers()
    return @async while !interrupt_flag.value
        try
            sleep(0.1)
        catch e
            if e isa InterruptException
                interrupt_flag.value = true
                println("\nReceived interrupt signal")
                break
            end
        end
    end
end

function main()
    signal_task = nothing
    try
        signal_task = setup_signal_handlers()
        config = load_environment()
        if (conn = SocketLogger.connect_to_python_socket_easy(config.host, config.port)) !== nothing
            SocketLogger.write_log_to_socket(conn, "Julia Worker started!\n")
            start_worker(conn, config.redis_host, config.redis_port)
        else
            println("Failed to establish connection")
        end
    catch e
        println("An error occurred: $e")
        global_interrupt_flag[] = true
        rethrow(e)
    finally
        global_interrupt_flag[] = true
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