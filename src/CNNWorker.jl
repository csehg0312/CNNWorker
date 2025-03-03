module CNNWorker
    using Distributed
    using SharedArrays
    using FFTW
    using LoopVectorization
    using DifferentialEquations
    using Images
    using FileIO
    using Base64
    using WebSockets
    using DistributedArrays
    using LinearAlgebra

    export solve_ode, setup_distributed_environment, ensure_workers

    # ------ Top-level @everywhere blocks ------
    # @everywhere begin
    #     using Distributed
    #     using SharedArrays
    #     using FFTW
    #     using LoopVectorization
    #     using DifferentialEquations
    #     using Images
    #     using FileIO
    #     using Base64
    #     using WebSockets
    #     using DistributedArrays
    #     using LinearAlgebra
    # end

    # @everywhere begin
    #     x = rand(10, 10)
    #     y = similar(x)
    #     for i in eachindex(x)
    #         y[i] = 0.5 * (abs(x[i] + 1) - abs(x[i] - 1))
    #     end
    # end

    # @everywhere begin
    #     import DifferentialEquations
    #     using LoopVectorization
    #     using SharedArrays
    #     using FFTW
    # end

    # ------ Activation Functions (Previously Activation module) ------
    function activation(x)
        return 0.5 * (abs(x + 1) - abs(x - 1))
    end

    function safe_activation(x)
        x_clamped = clamp(x, -1e6, 1e6)
        return activation(x_clamped)
    end

    function batch_activation!(output, input)
        @turbo for i in eachindex(input)
            output[i] = safe_activation(input[i])
        end
    end

    # ------ Convolution Functions (Previously LinearConvolution module) ------
    function fftconvolve2d(in1::Matrix{T}, in2::Matrix{T}; threshold=1e-10) where T<:Number
        s1 = size(in1)
        s2 = size(in2)

        # Calculate the padded size
        padded_size = (s1[1] + s2[1] - 1, s1[2] + s2[2] - 1)

        # Calculate the next power of 2 for efficient FFT
        next_power_of_2 = (2^ceil(Int, log2(padded_size[1])), 2^ceil(Int, log2(padded_size[2])))

        # Create padded arrays
        padded_in1 = zeros(Complex{Float64}, next_power_of_2)
        padded_in2 = zeros(Complex{Float64}, next_power_of_2)

        # Copy the original arrays into the padded arrays
        padded_in1[1:s1[1], 1:s1[2]] = in1
        padded_in2[1:s2[1], 1:s2[2]] = in2

        # Perform FFT on the padded arrays
        fft_in1 = fft(padded_in1)
        fft_in2 = fft(padded_in2)

        # Multiply the FFT results
        fft_result = fft_in1 .* fft_in2

        # Perform inverse FFT and apply thresholding
        result = real(ifft(fft_result))
        result[abs.(result) .< threshold] .= 0  # Set very small values to exact zero

        # Extract the valid part of the result with same dimensions as input
        start_row = div(s2[1] - 1, 2)
        start_col = div(s2[2] - 1, 2)
        return result[start_row+1:start_row+s1[1], start_col+1:start_col+s1[2]]
    end

    function parallel_fftconvolve2d(in1::Matrix{T}, in2::Matrix{T}; threshold=1e-10) where T<:Number
        s1 = size(in1)
        num_workers = nworkers()
        
        if num_workers <= 1 || s1[1] < 100  # For small matrices, use non-distributed version
            return fftconvolve2d(in1, in2; threshold=threshold)
        end
        
        # Split the input matrix into chunks for parallel processing
        chunk_size = div(s1[1], num_workers)
        result = SharedArray{Float64}(s1)
        
        @sync begin
            for (i, worker) in enumerate(workers())
                start_row = (i-1) * chunk_size + 1
                end_row = i == num_workers ? s1[1] : i * chunk_size
                
                @async remotecall_wait(compute_convolution_chunk!, worker, result, in1, in2, start_row, end_row, threshold)
            end
        end
        
        return Array(result)
    end
    
    function compute_convolution_chunk!(result, in1, in2, start_row, end_row, threshold)
        chunk = in1[start_row:end_row, :]
        conv_result = fftconvolve2d(chunk, in2; threshold=threshold)
        result[start_row:end_row, :] = conv_result
    end

    # ------ Utility Functions ------
    function partition(range, n)
        len = length(range)
        chunk_size = max(1, div(len, n))
        return [range[min(len, (i-1)*chunk_size+1):min(len, i*chunk_size)] for i in 1:n]
    end

    function f!(du, u, p, t)
        Ib, Bu, tempA, n, m, wsocket = p
        
        # Use a throttled logging to reduce communication overhead
        if round(t, digits=1) == t  # Log only at every 0.1 time unit
            WebSockets.write(wsocket, "Solving at $t time")
        end
        
        # Use shared memory for reshaping to avoid copying
        x_mat = reshape(u, n, m)
        
        # Apply activation function to x_mat
        batch_activation!(x_mat, x_mat)
        
        # Perform 2D convolution using FFT with distributed computation
        conv_result = parallel_fftconvolve2d(x_mat, tempA)
        
        # Compute the derivative du using vectorized operations
        @turbo for i in eachindex(du)
            du[i] = clamp(-u[i] + Ib + Bu[i] + conv_result[i], -1e6, 1e6)
        end
    end

    function ode_result_process(z, n, m)
        result = similar(z)
        
        # Process in parallel if the matrix is large enough
        if n*m > 10000 && nworkers() > 1
            result_shared = SharedArray{Float64}(size(z))
            result_shared[:] = z
            
            @sync @distributed for chunk in partition(1:length(z), nworkers())
                for i in chunk
                    result_shared[i] = safe_activation(z[i])
                end
            end
            out_l = reshape(Array(result_shared), n, m)
        else
            @turbo for i in eachindex(z)
                result[i] = safe_activation(z[i])
            end
            out_l = reshape(result, n, m)
        end

        min_val, max_val = extrema(out_l)
        if min_val != max_val
            @turbo for i in eachindex(out_l)
                out_l[i] = (out_l[i] - min_val) / (max_val - min_val)
            end
        else
        out_l .= 0.5
        end

        @turbo for i in eachindex(out_l)
            out_l[i] = round(UInt8, clamp(out_l[i], 0.0, 1.0) * 255)
        end

        return out_l
    end

    function cleanup_memory!(vars...)
        for var in vars
            try
                if var isa Array || var isa SharedArray || var isa DistributedArray || var isa IOBuffer
                    Base.finalize(var)
                end
            catch e
                @warn "Cleanup error: $e"
            end
            var = nothing
        end
        GC.gc(true)
    end

    function process_and_generate_image(z, n, m, wsocket)
        if length(z) == n*m
            out_l = reshape(copy(z), n, m)
        else
            out_l = reshape(copy(z[end]), n, m)
        end

        if n*m > 100000 && nworkers() > 1
            out_shared = SharedArray{Float64}(size(out_l))
            out_shared[:] = out_l[:]
            
            @sync @distributed for chunk in partition(1:length(out_shared), nworkers())
                for i in chunk
                    val = safe_activation(out_shared[i])
                    out_shared[i] = val > 0 ? 255.0 : 0.0
                end
            end
            out_l = reshape(Array(out_shared), n, m)
        else
            @turbo for i in eachindex(out_l)
                val = safe_activation(out_l[i])
                out_l[i] = val > 0 ? 255.0 : 0.0
            end
        end
        
        try
            binary_image = Gray.(out_l ./ 255)
            io = IOBuffer()
            FileIO.save(Stream(format"PNG", io), binary_image)
            binary_data = take!(io)
            img_base64 = base64encode(binary_data)
            image_packet = "data:image/png;base64,$img_base64"
            WebSockets.write(wsocket, image_packet)
            cleanup_memory!(binary_image, binary_data, img_base64, image_packet, io)
        catch e
            WebSockets.write(wsocket, "Image processing error: $e")
            cleanup_memory!(z, out_l)
        end
    end

    # ------ Main Functions ------
    function solve_ode(socket_conn, image::Matrix{Float64}, Ib::Float64, tempA::Matrix{Float64}, tempB::Matrix{Float64}, t_span::Vector{Float64}, initial_condition::Float64, wsocket)
        SocketLogger.write_log_to_socket(socket_conn, "Starting ODE solver with $(nprocs()) processes and $(nworkers()) workers\n")
        WebSockets.write(wsocket, "Started ODE solver with DifferentialEquations.jl distributed computation...")
        
        n, m = size(image)
        image_normalized = SharedArray{Float64}(n, m)
        
        if n*m > 10000 && nworkers() > 1
            @sync @distributed for chunk in partition(1:n*m, nworkers())
                for i in chunk
                    row, col = fldmod1(i, m)
                    image_normalized[row, col] = (image[row, col] / 127.5) - 1
                end
            end
        else
            @turbo for i in eachindex(image)
                image_normalized[i] = (image[i] / 127.5) - 1
            end
        end
        
        SocketLogger.write_log_to_socket(socket_conn, "Before Bu init")
        WebSockets.write(wsocket, "First convolution started with distributed computation")
        
        Bu = parallel_fftconvolve2d(Array(image_normalized), tempB)
        
        WebSockets.write(wsocket, "First convolution ended")
        SocketLogger.write_log_to_socket(socket_conn, "After Bu init")
        
        if n*m > 10000 && nworkers() > 1
            @sync @distributed for chunk in partition(1:n*m, nworkers())
                for i in chunk
                    image_normalized[i] *= initial_condition
                end
            end
        else
            @turbo for i in eachindex(image_normalized)
                image_normalized[i] *= initial_condition
            end
        end
        
        z0 = Array(image_normalized)
        params = (Ib, Bu, tempA, n, m, wsocket)

        SocketLogger.write_log_to_socket(socket_conn, "Before ODE problem")
        WebSockets.write(wsocket, "ODE Solver started with parallel settings!")
        
        prob = ODEProblem(f!, z0, (t_span[1], t_span[end]), params)
        
        if n*m > 50000
            sol = solve(
                prob, 
                CVODE_BDF(linear_solver=:GMRES), 
                reltol=1e-5, 
                abstol=1e-8, 
                maxiters=1000000,
                save_everystep=false,
                dt=min(0.1, (t_span[end]-t_span[1])/100),
                progress=true, 
                progress_steps=10,
                parallel_type=EnsembleThreads()
            )
        else
            alg = AutoTsit5(Rosenbrock23())
            sol = solve(
                prob, 
                alg, 
                reltol=1e-5, 
                abstol=1e-8,
                maxiters=1000000,
                save_everystep=false,
                dt=min(0.1, (t_span[end]-t_span[1])/100),
                progress=true, 
                progress_steps=10
            )
        end
        
        SocketLogger.write_log_to_socket(socket_conn, "After ODE problem")
        WebSockets.write(wsocket, "ODE solved successfully")

        process_and_generate_image(sol[end], n, m, wsocket)
        cleanup_memory!(prob, sol, Bu, z0, image_normalized, params)
    end

    function ensure_workers(min_workers=4)
        current = nworkers()
        if current < min_workers
            addprocs(min_workers - current)
        end
        return nworkers()
    end

    function setup_distributed_environment(num_workers=Sys.CPU_THREADS-1)
        if nprocs() == 1
            workers_count = ensure_workers(num_workers)
            @info "Initialized $workers_count workers for distributed computation"
            
            LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
            @info "BLAS using $(LinearAlgebra.BLAS.get_num_threads()) threads"
            
            FFTW.set_num_threads(Threads.nthreads())
            @info "FFTW using $(FFTW.get_num_threads()) threads"
        end
        
        return nprocs(), nworkers(), Threads.nthreads()
    end
end