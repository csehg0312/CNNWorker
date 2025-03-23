module SocketLogger
using Sockets
using ..CNNWorker

export connect_to_python_socket, write_log_to_socket, connect_to_python_socket_easy

function connect_to_python_socket(host::String, port::Int,  max_attempts::Int=5, retry_delay::Float64=4.0)
    for attempt in 1:max_attempts
        try
            sock = Sockets.connect(host, port)
            return sock
        catch e
            if isa(e, Base.IOError) && attempt < max_attempts
                @warn "Connection attempt $attempt failed. Retrying in $retry_delay seconds..."
                sleep(retry_delay)
            else
                rethrow(e)
            end
        end
    end
end

function connect_to_python_socket_easy(host::String, port::Int)
    try
        sock = connect(host, port)
        println("Type of sock: ", typeof(sock))
        return sock
        # println("Connected successfully!")
        # write(sock, "Hello from Julia!")
        # close(sock)
    catch e
        println("Failed to connect: $e, on port $port to host $host")
    end
end

function write_log_to_socket(s::IO, log_message::String)
    write(s, log_message)
    flush(s)
end

end
