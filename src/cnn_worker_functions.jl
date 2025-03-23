# Define activation functions
activation(x) = 0.5 * (abs(x + 1) - abs(x - 1))
safe_activation(x) = activation(clamp(x, -1e6, 1e6))

function batch_activation!(output, input)
    @turbo for i in eachindex(input)
        output[i] = safe_activation(input[i])
    end
end

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

# function process_and_generate_image(z, n, m, wsocket)
#     out_l = reshape(z, n, m)
#     @turbo for i in eachindex(out_l)
#         out_l[i] = round(UInt8, clamp(out_l[i], 0.0, 1.0) * 255)
#     end

#     binary_image = Gray.(out_l ./ 255)
#     io = IOBuffer()
#     FileIO.save(Stream(format"PNG", io), binary_image)
#     binary_data = take!(io)
#     img_base64 = base64encode(binary_data)
#     image_packet = "data:image/png;base64,$img_base64"
#     WebSockets.write(wsocket, image_packet)
# end

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

    catch e
        println("Error $e")
    end
end
