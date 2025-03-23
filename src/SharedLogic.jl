module SharedLogic
using Redis, JSON, WebSockets, SharedArrays, LinearAlgebra
include("SocketLogger.jl")
include("SharedState.jl")
end
