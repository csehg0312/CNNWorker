module CNNWorker

using FFTW, LoopVectorization, Sockets, Redis, DotEnv
using SharedArrays, Distributed, DistributedArrays, LinearAlgebra
using Images, FileIO, Base64, WebSockets, DifferentialEquations

# Include dependencies
for file in ["cnn_worker_functions.jl", "RedisQueueWatcher.jl","SocketLogger.jl", "ODESolver.jl"]
    include(file)
end

# Global variable to track interrupt state
const global_interrupt_flag = Ref(false)

# Load environment variables
function load_environment()
    env_path = joinpath(dirname(dirname(@__DIR__)), ".env")
    DotEnv.load!(env_path)

    return (
        host = ENV["JULIA_PYTHON_HOST"],
        port = parse(Int, ENV["JULIA_PYTHON_PORT3"]),
        redis_host = ENV["REDIS_HOST"],
        redis_port = parse(Int, ENV["REDIS_PORT"])
    )
end

"""
    setup_distributed_environment(num_workers=Sys.CPU_THREADS-1)

Sets up distributed computing environment with the specified number of worker processes.
Returns a tuple of (total processes, worker processes, threads per process)
"""
function setup_distributed_environment(num_workers=Sys.CPU_THREADS-1)
    if nprocs() == 1
        addprocs(num_workers)

        # Configure thread safety
        LinearAlgebra.BLAS.set_num_threads(1)
        FFTW.set_num_threads(1)

        # Load packages on workers using @everywhere
        @everywhere using SharedArrays, FFTW, LoopVectorization, DifferentialEquations
        @everywhere using Images, FileIO, Base64, WebSockets, DistributedArrays, LinearAlgebra
        @everywhere include("cnn_worker_functions.jl")
    end

    return nprocs(), nworkers(), Threads.nthreads()
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
        # Setup distributed environment first
        total_procs, worker_procs, threads_per_proc = setup_distributed_environment()
        SocketLogger.write_log_to_socket(socket_conn,
            "Distributed environment setup: $total_procs processes, $worker_procs workers, $threads_per_proc threads per process\n")

        redis_client = Redis.connect(redis_host, redis_port)

        Base.sigatomic_begin()
        RedisQueueWatcher.watch_redis_queue(redis_client, "queue:task_queue", socket_conn)
        Base.sigatomic_end()
    catch e
         if e isa InterruptException
            SocketLogger.write_log_to_socket(socket_conn,
                "Received interrupt signal. Shutting down gracefully...\n")
        else
            SocketLogger.write_log_to_socket(socket_conn, "An error occurred: $e\n")
            SocketLogger.write_log_to_socket(socket_conn,
                "Backtrace: $(catch_backtrace())\n")
        end
        rethrow(e)
    finally
        # Cleanup worker processes before shutting down
        if nworkers() > 0
            SocketLogger.write_log_to_socket(socket_conn, "Removing worker processes...\n")
            rmprocs(workers())
        end
        handle_shutdown(redis_client, socket_conn)
    end
end

function setup_signal_handlers()
    signal_task = @async begin
        try
            while !global_interrupt_flag[]
                try
                    sleep(0.1)
                catch e
                    if e isa InterruptException
                        global_interrupt_flag[] = true
                        println("\nReceived interrupt signal")
                        break
                    else
                        rethrow(e)
                    end
                end
            end
        catch e
            println("Signal handler error: $e")
        end
    end
 return signal_task
end

# Main execution
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

# Run the worker
main()

end # module
