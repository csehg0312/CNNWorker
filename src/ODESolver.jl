module ODESolver

using DifferentialEquations, WebSockets, Base64
using SharedArrays, Distributed, DistributedArrays
using ..SocketLogger

export solve_ode, solve_chunk

function solve_ode(socket_conn, image_matrix, Ib, feedbackA_matrix, controlB_matrix, t_span, initialCondition, ws)
    SocketLogger.write_log_to_socket(socket_conn, "Starting CPU-based ODE solution\n")

    n, m = size(image_matrix)
    SocketLogger.write_log_to_socket(socket_conn, "Matrix dimensions: $n x $m\n")

    # Check if we should use distributed processing
    if nworkers() > 1
        SocketLogger.write_log_to_socket(socket_conn, "Using distributed processing with $(nworkers()) workers\n")
        solve_ode_distributed(socket_conn, image_matrix, Ib, feedbackA_matrix, controlB_matrix, t_span, initialCondition, ws)
    else
        solve_ode_sequential(socket_conn, image_matrix, Ib, feedbackA_matrix, controlB_matrix, t_span, initialCondition, ws)
    end
end

function solve_ode_sequential(socket_conn, image_matrix, Ib, feedbackA_matrix, controlB_matrix, t_span, initialCondition, ws)
    n, m = size(image_matrix)
    u0 = fill(initialCondition, n*m)

    # Setup parameters for ODE
    p = (Ib, controlB_matrix, feedbackA_matrix, n, m)

    # Create the ODE problem
    prob = ODEProblem(f!, u0, (t_span[1], t_span[end]), p)

    # Solve ODE with specified timepoints
    SocketLogger.write_log_to_socket(socket_conn, "Solving ODE sequentially...\n")
    sol = solve(prob, Tsit5(), saveat=t_span)

    # Process results for each time point
    SocketLogger.write_log_to_socket(socket_conn, "Processing solution results...\n")
    for (t_idx, t) in enumerate(t_span)
        z = sol(t)

        # Send progress update
        if mod(t_idx, 5) == 0 || t_idx == length(t_span)
            progress = round(t_idx / length(t_span) * 100, digits=1)
            WebSockets.write(ws, "Progress: $progress%")
        end

        # Process and send the image at specified intervals
        if mod(t_idx, 10) == 0 || t_idx == length(t_span)
            process_and_generate_image(z, n, m, ws)
        end
    end

    WebSockets.write(ws, "Computation complete!")
    SocketLogger.write_log_to_socket(socket_conn, "ODE solution completed\n")
end

function solve_ode_distributed(socket_conn, image_matrix, Ib, feedbackA_matrix, controlB_matrix, t_span, initialCondition, ws)
    n, m = size(image_matrix)

    # Determine how to divide the work
    num_workers = nworkers()
    chunk_size = div(n, num_workers)

    # Distribute the matrices
    d_image = DistributedArrays.distribute(image_matrix, procs=workers(), dist=[num_workers, 1])
    d_feedbackA = DistributedArrays.distribute(feedbackA_matrix, procs=workers(), dist=[num_workers, 1])
    d_controlB = DistributedArrays.distribute(controlB_matrix, procs=workers(), dist=[num_workers, 1])

    SocketLogger.write_log_to_socket(socket_conn, "Distributed arrays created\n")

    # Create a SharedArray to store results
    final_results = SharedArray{Float64}(n, m, length(t_span))
    # Process in parallel
    @sync begin
        for (i, worker) in enumerate(workers())
            start_row = (i-1) * chunk_size + 1
            end_row = i == num_workers ? n : i * chunk_size

            @async remotecall_wait(solve_chunk!, worker, final_results,
                                  image_matrix[start_row:end_row, :],
                                  Ib,
                                  feedbackA_matrix[start_row:end_row, :],
                                  controlB_matrix[start_row:end_row, :],
                                  t_span, initialCondition,
                                  start_row, end_row)
        end
    end

    SocketLogger.write_log_to_socket(socket_conn, "All chunks processed\n")

    # Process results for each time point
    for (t_idx, t) in enumerate(t_span)
        # Send progress update
        if mod(t_idx, 5) == 0 || t_idx == length(t_span)
            progress = round(t_idx / length(t_span) * 100, digits=1)
            WebSockets.write(ws, "Progress: $progress%")
        end

        # Process and send the image at specified intervals
        if mod(t_idx, 10) == 0 || t_idx == length(t_span)
            z = final_results[:, :, t_idx]
            process_and_generate_image(z, n, m, ws)
        end
    end

    WebSockets.write(ws, "Computation complete!")
    SocketLogger.write_log_to_socket(socket_conn, "Distributed ODE solution completed\n")
end
# Function that will be called on worker processes to solve a chunk of the problem
function solve_chunk!(result_array, image_chunk, Ib, feedbackA_chunk, controlB_chunk, t_span, initialCondition, start_row, end_row)
    n_chunk, m = size(image_chunk)
    u0 = fill(initialCondition, n_chunk*m)

    # Setup parameters for ODE
    p = (Ib, controlB_chunk, feedbackA_chunk, n_chunk, m)

    # Create the ODE problem
    prob = ODEProblem(f!, u0, (t_span[1], t_span[end]), p)

    # Solve ODE with specified timepoints
    sol = solve(prob, Tsit5(), saveat=t_span)

    # Store results in the shared array
    for (t_idx, t) in enumerate(t_span)
        z = sol(t)
        result_array[start_row:end_row, :, t_idx] = reshape(z, n_chunk, m)
    end

    return nothing
end
# This version is for external calls from RedisQueueWatcher when distributed processing is preferred
function solve_chunk(image_chunk, Ib, feedbackA_chunk, controlB_chunk, t_span, initialCondition)
    n_chunk, m = size(image_chunk)
    u0 = fill(initialCondition, n_chunk*m)

    # Setup parameters for ODE
    p = (Ib, controlB_chunk, feedbackA_chunk, n_chunk, m)

    # Create the ODE problem
    prob = ODEProblem(f!, u0, (t_span[1], t_span[end]), p)

    # Solve ODE with specified timepoints
    sol = solve(prob, Tsit5(), saveat=t_span)

    # Prepare results
    results = zeros(n_chunk, m, length(t_span))
    for (t_idx, t) in enumerate(t_span)
        z = sol(t)
        results[:, :, t_idx] = reshape(z, n_chunk, m)
    end

    return results
end

end # module
