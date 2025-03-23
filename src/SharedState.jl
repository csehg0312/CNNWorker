module SharedState
export InterruptFlag

mutable struct InterruptFlag
    value::Bool
end

function InterruptFlag()
    return InterruptFlag(false)
end

end # module