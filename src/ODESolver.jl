module ODESolver

using DifferentialEquations, WebSockets, Base64
using SharedArrays, Distributed, DistributedArrays
using FFTW, LoopVectorization, FileIO, Images
using Base: IOBuffer, @async, @sync

# Define activation functions
activation(x) = 0.5 * (abs(x + 1) - abs(x - 1))
safe_activation(x) = activation(clamp(x, -1e6, 1e6))

function batch_activation!(output, input)
    @turbo for i in eachindex(input)
        output[i] = safe_activation(input[i])
    end
end

# Convolution functions
function fftconvolve2d(in1::Matrix{T}, in2::Matrix{T}; threshold=1e-10) where T<:Number
    s1 = size(in1)
    s2 = size(in2)
    padded_size = (s1[1] + s2[1] - 1, s1[2] + s2[2] - 1)
    next_power_of_2 = (2^ceil(Int, log2(padded_size[1])), 2^ceil(Int, log2(padded_size[2])))

    padded_in1 = zeros(Complex{Float64}, next_power_of_2)
    padded_in2 = zeros(Complex{Float64}, next_power_of_2)
    padded_in1[1:s1[1], 1:s1[2]] = in1
    padded_in2[1:s2[1], 1:s2[2]] = in2

    fft_in1 = fft(padded_in1)
    fft_in2 = fft(padded_in2)
    fft_result = fft_in1 .* fft_in2
    result = real(ifft(fft_result))
    result[abs.(result) .< threshold] .= 0
    return result
end

function parallel_fftconvolve2d(in1::Matrix{T}, in2::Matrix{T}; threshold=1e-10) where T<:Number
    s1 = size(in1)
    num_workers = nworkers()

    if num_workers <= 1 || s1[1] < 100
        return fftconvolve2d(in1, in2; threshold=threshold)
    end

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
    
    # Crop the convolution result to match the destination dimensions
    cropped_result = conv_result[1:end_row-start_row+1, 1:size(result, 2)]
    
    # Assign the cropped result to the SharedArray
    result[start_row:end_row, :] = cropped_result
    return nothing
end

# ODE function
function f!(du, u, p, t)
    Ib, Bu, tempA, n, m = p
    x_mat = reshape(u, n, m)
    activated_x = similar(x_mat)
    batch_activation!(activated_x, x_mat)
    conv_result = parallel_fftconvolve2d(activated_x, tempA)

    @turbo for i in eachindex(du)
        du[i] = clamp(-u[i] + Ib + Bu[i] + conv_result[i], -1e6, 1e6)
    end
end

# Image processing function
function process_and_generate_image(z, n, m, wsocket)
    out_l = reshape(z, n, m)

    # Threshold values and check for NaN/Inf
    out_l .= ifelse.(isnan.(out_l) .| isinf.(out_l), 0.0, ifelse.(out_l .> 0, 255.0, 0.0))

    try
        # Ensure all values are valid
        clamped_out = clamp.(out_l, 0.0, 255.0)

        # Convert to image
        binary_image = Gray.(clamped_out ./ 255)

        # First step: Fix horizontal mirroring by flipping the original image
        # This reverses the order of columns (horizontal flip)
        unmirrored_image = reverse(binary_image, dims=2)

        # Now rotate the unmirrored image
        # We'll use rotl90 instead of imrotate to avoid interpolation issues
        rotated_image = rotl90(unmirrored_image)

        # Verify rotated image validity
        if any(isnan, rotated_image) || any(isinf, rotated_image)
            WebSockets.write(wsocket, "Invalid pixel values after rotation")
            return
        end

        # Continue with encoding
        io_rotated = IOBuffer()
        FileIO.save(Stream(format"PNG", io_rotated), rotated_image)
        binary_data = take!(io_rotated)
        img_base64 = base64encode(binary_data)
        image_packet = "data:image/png;base64,$img_base64"
        WebSockets.write(wsocket, image_packet)
    catch e
        println("Error $e")
    end
end

# Main ODE solver function
function solve_ode(socket_conn, image_matrix, Ib, feedbackA_matrix, controlB_matrix, t_span, initialCondition, ws)
    n, m = size(image_matrix)
    u0 = fill(initialCondition, n * m)

    # Ensure feedbackA and controlB are padded to match image_matrix dimensions
    padded_feedbackA = pad_to_size(feedbackA_matrix, (n, m))
    padded_controlB = pad_to_size(controlB_matrix, (n, m))

    p = (Ib, padded_controlB, padded_feedbackA, n, m)
    prob = ODEProblem(f!, u0, (t_span[1], t_span[end]), p)

    sol = solve(prob, Tsit5(), saveat=t_span)

    for (t_idx, t) in enumerate(t_span)
        z = sol(t)

        if mod(t_idx, 5) == 0 || t_idx == length(t_span)
            progress = round(t_idx / length(t_span) * 100, digits=1)
            WebSockets.write(ws, "Progress: $progress%")
        end

        if mod(t_idx, 10) == 0 || t_idx == length(t_span)
            process_and_generate_image(z, n, m, ws)
        end
    end

    WebSockets.write(ws, "Computation complete!")
end

# Helper function to pad smaller matrices to match target size
function pad_to_size(matrix, target_size)
    padded = zeros(target_size)
    n, m = size(matrix)
    padded[1:n, 1:m] = matrix
    return padded
end

end # module