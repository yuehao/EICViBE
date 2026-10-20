# The Julia side of the TrackPad engine (eicvibe/engines/trackpad.py, ADR 0006).
#
# Python describes each TrackPad element as a kind code, a name and a row of
# numbers whose columns are named in a header, so the conversion of the
# lattice stays in Python and this module only calls TrackPad's constructors
# and computations. Coordinates are TrackPad's: (x, px, y, py, z, dE) with
# z = s/beta0 - c t and dE = (E - E0)/(P0 c).

module EICViBETrackPad

using LinearAlgebra
using StaticArrays
using TrackPad

const KINDS = (
    :drift, :marker, :quadrupole, :sextupole, :octupole, :bend, :multipole,
    :corrector, :solenoid, :rfcavity, :patch, :translation, :exactbend,
)

struct Row{V<:AbstractVector}
    values::V
    columns::Dict{Symbol,Int}
end

Base.getindex(row::Row, name::Symbol) = Float64(row.values[row.columns[name]])

header(columns::AbstractString) = Dict(Symbol(c) => i for (i, c) in enumerate(split(columns, ',')))

polynom_b(r::Row) = SVector(r[:b0], r[:b1], r[:b2], r[:b3])
polynom_a(r::Row) = SVector(r[:a0], r[:a1], r[:a2], r[:a3])
order(r::Row) = round(Int, r[:order])
steps(r::Row) = round(Int, r[:steps])

"""Entrance and exit maps of a displaced, pitched, yawed and rolled element (AT's T1, T2, R1, R2).

The body's centre is offset by (dx, dy) and its axis turned about the centre by PALS's right-handed
pitch (x_rot: the exit towards -y) and yaw (y_rot: the exit towards +x); to first order in the
angles the entrance face is then half a length back along the turned axis, and the particle enters
at the angle the axis makes. The roll turns the body about its own axis after that.
"""
function placement(r::Row)
    dx, dy, roll, pitch, yaw = r[:dx], r[:dy], r[:roll], r[:pitch], r[:yaw]
    half = r[:L] / 2
    moved = !(iszero(dx) && iszero(dy) && iszero(pitch) && iszero(yaw))
    shift = moved ? (t1=SVector(-dx + yaw * half, -yaw, -dy - pitch * half, pitch, 0.0, 0.0),
                     t2=SVector(dx + yaw * half, yaw, dy - pitch * half, -pitch, 0.0, 0.0)) : (;)
    iszero(roll) && return shift
    c, s = cos(roll), sin(roll)
    r1 = @SMatrix [c 0 s 0 0 0; 0 c 0 s 0 0; -s 0 c 0 0 0; 0 -s 0 c 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]
    return (; shift..., r1=r1, r2=SMatrix{6,6}(transpose(r1)))
end

function element(kind::Symbol, name::Symbol, r::Row, beam::Beam, rad::Integer=0)
    if kind === :drift
        return Drift(r[:L]; name=name)
    elseif kind === :marker
        return Marker(; name=name)
    elseif kind === :quadrupole
        if r[:closed] == 1 && CLOSED_QUADRUPOLES  # its gradient in closed form, as Xsuite's thick quadrupole
            return LBend(r[:L], 0.0; K=r[:b1], name=name, rad=Int(rad), placement(r)...)
        end
        return Quadrupole(r[:L], r[:b1]; name=name, polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                          max_order=order(r), num_int_steps=steps(r), rad=Int(rad), placement(r)...)
    elseif kind === :sextupole
        return Sextupole(r[:L], 2 * r[:b2]; name=name, polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                         max_order=order(r), num_int_steps=steps(r), rad=Int(rad), placement(r)...)
    elseif kind === :octupole
        return Octupole(r[:L], 6 * r[:b3]; name=name, polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                        max_order=order(r), num_int_steps=steps(r), rad=Int(rad), placement(r)...)
    elseif kind === :bend
        return SBend(r[:L], r[:angle], r[:e1], r[:e2]; name=name,
                     fint1=r[:fint1], fint2=r[:fint2], gap=r[:gap],
                     polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                     max_order=order(r), num_int_steps=steps(r), rad=Int(rad), placement(r)...)
    elseif kind === :exactbend
        return ExactSBend(r[:L], r[:angle], r[:e1], r[:e2]; name=name, gk=r[:gk],
                          polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                          max_order=order(r), num_int_steps=steps(r), rad=Int(rad), placement(r)...)
    elseif kind === :multipole
        return ThinMultipole(0.0, polynom_a(r), polynom_b(r); name=name, max_order=order(r), placement(r)...)
    elseif kind === :corrector
        return Corrector(0.0, r[:xkick], r[:ykick]; name=name, placement(r)...)
    elseif kind === :solenoid
        return Solenoid(r[:L], r[:ks]; name=name, placement(r)...)
    elseif kind === :rfcavity
        return RFCavity(r[:L], r[:volt], r[:freq], 0.0; name=name, h=r[:harmon], philag=r[:philag],
                        energy=beam.energy, charge=abs(beam.charge))
    elseif kind === :patch
        return Patch(; name=name, x_pitch=r[:x_pitch], y_pitch=r[:y_pitch], tilt=r[:tilt], t_offset=r[:t_offset])
    elseif kind === :translation
        return Translation(0.0; name=name, dx=r[:shift_x], dy=r[:shift_y], ds=r[:shift_s])
    end
    throw(ArgumentError("unknown element kind $kind"))
end

"""The elements of `kinds` (codes into KINDS), `names` (newline-separated) and rows of `values`."""
function elements(kinds::AbstractVector, names::AbstractString, columns::AbstractString,
                  values::AbstractMatrix, beam::Beam)
    index = header(columns)
    labels = split(names, '\n')
    out = Vector{AbstractElement}(undef, length(kinds))
    for i in eachindex(out)
        out[i] = element(KINDS[Int(kinds[i]) + 1], Symbol(labels[i]), Row(view(values, i, :), index), beam)
    end
    return out
end

function lattice(kinds, names, columns, values, beam::Beam, periodic::Bool)
    return Lattice(elements(kinds, names, columns, values, beam); name=:branch, periodic=periodic)
end

"""Replace elements from `first` on (1-based) with new ones, in place."""
function setelements!(lat::Lattice, first::Integer, kinds, names, columns, values, beam::Beam)
    new = elements(kinds, names, columns, values, beam)
    lat.elements[first:first + length(new) - 1] .= new
    return nothing
end

beam(pc::Real, mass::Real, charge::Real) = Beam(; pc=Float64(pc), mass=Float64(mass), charge=Float64(charge))

"""Elements `first` to `last` (1-based, inclusive) as a line: part of a branch, passed once."""
segment(lat::Lattice, first::Integer, last::Integer) =
    Lattice(lat.elements[first:last]; name=lat.name, periodic=false)

"""The lattice with its RF cavities as drifts: the 4D (frozen longitudinal) model."""
function transverse(lat::Lattice)
    any(e -> e isa RFCavity, lat.elements) || return lat
    plain = AbstractElement[e isa RFCavity ? Drift(e.L; name=e.name) : e for e in lat.elements]
    return Lattice(plain; name=lat.name, periodic=lat.periodic)
end

deltae(delta::Real, b::Beam) = TrackPad.deltae_from_deltap(Float64(delta), b.beta)

"""Whether this TrackPad's LBend radiates: a pure quadrupole is then an LBend of angle 0, its gradient in
closed form (the expanded Hamiltonian, as Xsuite's thick quadrupole map); otherwise it is integrated."""
const CLOSED_QUADRUPOLES = hasfield(LBend, :rad)

"""Whether this TrackPad radiates in its magnets: `rad` 1 damps, 2 also excites (Sands' photon diffusion,
Gaussian, its random numbers keyed by seed, turn, element, particle and slice)."""
const RADIATES = isdefined(TrackPad, :QUANTUM_CONSTANT_EV_M)

"""Whether this TrackPad's `track!` observes the bunch after chosen elements (`observe`, `observed`): a
ring is then tracked whole, its bunch spread over Julia's threads, and its BPMs read in the same pass;
otherwise it is cut into stretches between BPMs, tracked one after another on one thread."""
const OBSERVES = hasmethod(track!, Tuple{Matrix{Float64},Lattice,Beam}, (:observe, :observed))

"""Whether tracking may spread a bunch over Julia's threads (`PYTHON_JULIACALL_THREADS`): off to compare."""
const THREADED = Ref(true)

"""A magnet radiating as `mode` says (TrackPad's `rad`); other elements as they are."""
function radiating(e::AbstractElement, mode::Integer)
    T = typeof(e)
    (hasfield(T, :rad) && !(e isa Wiggler) && getfield(e, :rad) != mode) || return e
    return T((f === :rad ? Int(mode) : getfield(e, f) for f in fieldnames(T))...)
end

"""The lattice with its magnets radiating as `mode` says."""
radiating(lat::Lattice, mode::Integer) =
    Lattice(AbstractElement[radiating(e, mode) for e in lat.elements]; name=lat.name, periodic=lat.periodic)

"""Segment `k`'s random numbers' seed: TrackPad keys them by the element's place in what it tracks, which
starts again in each segment."""
segment_seed(seed::Integer, k::Integer) = xor(UInt64(seed), UInt64(k) * 0x9e3779b97f4a7c15)

"""One pass of the bunch `r` through `segment`, the `k`th, on turn `t`: radiating with its magnets."""
function pass_segment!(r, segment::Lattice, b::Beam, lost, t::Integer, k::Integer, seed::Integer)
    isempty(segment.elements) && return nothing
    if RADIATES
        linepass!(r, segment, b, lost; turn=t, radiation_seed=segment_seed(seed, k))
    else
        linepass!(r, segment, b, lost)
    end
    return nothing
end

"""A ring's 6D closed orbit with its magnets radiating (`lat`, `rad` 1): where the energy loss puts the
particle, which TrackPad's own finders, ignoring `rad`, do not see. Newton's method on one turn's
tracking, from the orbit without the radiation."""
function closed_orbit_radiating(lat::Lattice, b::Beam; tolerance=1e-15, iterations=40, step=1e-9)
    x = SVector{6,Float64}(find_closed_orbit_6d(lat, b)...)
    unit(j) = SVector{6,Float64}(ntuple(k -> k == j ? step : 0.0, 6))
    for _ in 1:iterations
        f = track(lat, x, b)
        any(isnan, f) && error("the closed orbit's particle is lost in one turn")
        J = hcat(((track(lat, x + unit(j), b) - f) / step for j in 1:6)...)
        dx = (J - I) \ (f - x)
        x = x - SVector{6,Float64}(dx)
        maximum(abs, dx) < tolerance && break
    end
    return collect(x)
end

"""Coordinates (6 x elements+1) at the entrance and after every element, from `start`."""
function along(lat::Lattice, b::Beam, start::AbstractVector; radiating::Bool=false)
    r = SVector{6,Float64}(start...)
    out = Matrix{Float64}(undef, 6, length(lat.elements) + 1)
    out[:, 1] = r
    inverse_beta = 1 / b.beta
    for (i, e) in enumerate(lat.elements)
        r = radiating ? TrackPad._pass_tracking(e, r, b, 0, 0, i, 1) : TrackPad.pass!(e, r, inverse_beta)
        out[:, i + 1] = r
    end
    return out
end

"""A ring's closed orbit: 4D at momentum `delta`, or 6D (the RF sets the momentum)."""
function closed_orbit(lat::Lattice, b::Beam, delta::Real, six::Bool)
    if six
        return collect(find_closed_orbit_6d(lat, b; x0=SVector(0.0, 0.0, 0.0, 0.0, 0.0, deltae(delta, b))))
    end
    dE = deltae(delta, b)
    x = find_closed_orbit_4d(lat, b; dp=dE, wrt=:deltae)
    return [x[1], x[2], x[3], x[4], 0.0, dE]
end

"""Entrance optics; phases in radians, dispersion with respect to delta."""
entrance(bx, ax, mux, dx, dpx, by, ay, muy, dy, dpy) =
    optics4DUC(optics2D(bx, ax, mux, dx, dpx), optics2D(by, ay, muy, dy, dpy))

"""Optics (rows s, betx, alfx, bety, alfy, mux, muy, dx, dpx, dy, dpy), the element each column
after the first ends, and a ring's summary.

The maps are TPSA (truncated power series): closed orbit, Jacobians and
chromaticity are exact derivatives, with no finite-difference steps; a lattice
with an LBend, which has no series map, takes finite differences of tracking
(TrackPad's `:fd`: tunes to 1e-12, chromaticity to 1e-6). With
`inside`, magnets are sampled at their integrator steps (one piece per step,
each integrated in one step), which adds points inside them and changes no
result.
"""
function optics(lat::Lattice, b::Beam, start, reference::AbstractVector, inside::Bool=false)
    ref = SVector{6,Float64}(reference...)
    sampling = (sample_integrator_steps=inside,)
    method = any(e -> e isa LBend, lat.elements) ? :fd : :tpsa
    tw = start === nothing ? twiss(lat, b; reference=ref, method=method, sampling...) :
         twiss(lat, b; entrance=start, reference=ref, method=method, sampling...)
    columns = permutedims(hcat(tw.s, tw.betax, tw.alphax, tw.betay, tw.alphay, tw.mux, tw.muy,
                               tw.dx, tw.dpx, tw.dy, tw.dpy))
    owner = inside ? owners(lat, refine_lattice(lat; sampling...)) : collect(1:length(lat.elements))
    return (columns=columns, owner=owner, tunex=tw.tunex, tuney=tw.tuney,
            chromx=tw.chromx, chromy=tw.chromy, alphac=tw.alphac)
end

"""For every piece of a sampled lattice, the element of `lat` it belongs to.

Pieces keep their element's name, and every element the adapter builds has its own.
"""
function owners(lat::Lattice, fine::Lattice)
    out = Vector{Int}(undef, length(fine.elements))
    j = 1
    for (p, piece) in enumerate(fine.elements)
        while piece.name !== lat.elements[j].name
            j += 1
        end
        out[p] = j
    end
    return out
end

"""Track `coords` (particles x 6) `turns` times, recording (x, px, y, py) after the elements in `cuts`.

Returns the record (cuts x turns x particles x 4), NaN once a particle is
lost, and the lost flags.
"""
function observe(lat::Lattice, b::Beam, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector;
                 rad::Integer=0, seed::Integer=0)
    r = Matrix{Float64}(coords)
    n = size(r, 1)
    lost = zeros(Int, n)
    iszero(rad) || (lat = radiating(lat, rad))
    bounds = [0; Int.(cuts); length(lat.elements)]
    segments = [Lattice(lat.elements[bounds[k] + 1:bounds[k + 1]]) for k in 1:length(bounds) - 1]
    record = fill(NaN, length(cuts), turns, n, 4)
    for t in 1:turns
        for (k, segment) in enumerate(segments)
            pass_segment!(r, segment, b, lost, t, k, seed)
            k > length(cuts) && continue
            for p in 1:n
                iszero(lost[p]) || continue
                record[k, t, p, :] .= @view r[p, 1:4]
            end
        end
    end
    return record, lost
end

"""Track `coords` (particles x 6) `turns` times and average them after the elements in `cuts`, as BPMs do.

Returns the centroids x and y (cuts x turns, NaN where no particle is left), the
number of particles averaged, the final coordinates and the lost flags.
"""
function centroids(lat::Lattice, b::Beam, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector;
                   rad::Integer=0, seed::Integer=0)
    r, lost, through, x, y, count = _setup(lat, coords, turns, cuts, rad)
    _turns!(through, r, b, lost, x, y, count, 1:turns, cuts, seed)
    return x, y, count, r, lost
end

"""
    centroids(lat, b, coords, turns, cuts, first, count, kinds, names, columns, base, group, ramp, delta, fraction[, momentum])

As `centroids`, while settings move turn by turn. Group `g` is the pieces `first[g]` to
`first[g] + count[g] - 1` of `lat` (1-based), one occurrence; its rows are the next `count[g]` rows of
`base` (and of `kinds`, `names`). A contribution `c` adds `fraction[ramp[c], t] * delta_c` to group
`group[c]`'s rows, `delta_c` being its next `count[group[c]]` rows of `delta`. Before each turn whose
fractions changed, the group's elements are built anew from those rows: element inputs are linear in
the settings, so a magnet follows its supply's ramp exactly, turn by turn, and nothing steps.

With `momentum` (P0 c in eV, a value per turn; `b` is the first turn's reference), the reference moves:
before each turn where it changes, the bunch is put about the new one (`rereference!`) and every group
is built anew with it. Magnets' strengths are normalized, so they follow the reference by
themselves; the RF cavities, whose kick TrackPad normalizes by the energy they are built with, are
groups of their own, so that they follow it too.
"""
function centroids(lat::Lattice, b::Beam, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector,
                   first::AbstractVector, count::AbstractVector, kinds::AbstractVector, names::AbstractString,
                   columns::AbstractString, base::AbstractMatrix, group::AbstractVector, ramp::AbstractVector,
                   delta::AbstractMatrix, fraction::AbstractMatrix, momentum=nothing;
                   rad::Integer=0, seed::Integer=0)
    r, lost, through, x, y, counts = _setup(lat, coords, turns, cuts, rad)
    place = _places(through, cuts, length(lat.elements))
    index = header(columns)
    labels = split(names, '\n')
    groups = length(first)
    rows_at = cumsum([1; Int.(count)])
    contributions = [Int[] for _ in 1:groups]
    delta_at = zeros(Int, length(group))
    offset = 1
    for c in eachindex(group)
        g = Int(group[c])
        push!(contributions[g], c)
        delta_at[c] = offset
        offset += Int(count[g])
    end
    rows = Matrix{Float64}(undef, maximum(Int.(count); init=0), size(base, 2))
    beam = b
    for t in 1:turns
        moved = false
        if momentum !== nothing && t > 1 && momentum[t] != momentum[t - 1]
            next = Beam(; pc=Float64(momentum[t]), mass=b.mass, charge=b.charge)
            rereference!(r, lost, beam, next)
            beam, moved = next, true
        end
        for g in 1:groups
            if t > 1 && !moved &&
               all(fraction[Int(ramp[c]), t] == fraction[Int(ramp[c]), t - 1] for c in contributions[g])
                continue
            end
            p = Int(count[g])
            at = rows_at[g]
            part = view(rows, 1:p, :)
            part .= view(base, at:at + p - 1, :)
            for c in contributions[g]
                part .+= fraction[Int(ramp[c]), t] .* view(delta, delta_at[c]:delta_at[c] + p - 1, :)
            end
            for j in 1:p
                e = element(KINDS[Int(kinds[at + j - 1]) + 1], Symbol(labels[at + j - 1]),
                            Row(view(part, j, :), index), beam, rad)
                _place!(through, place, Int(first[g]) + j - 1, e)
            end
        end
        _turns!(through, r, beam, lost, x, y, counts, t:t, cuts, seed)
    end
    return x, y, counts, r, lost
end

"""Put the bunch `r` (particles x 6, TrackPad's coordinates) about the reference `new` instead of `old`:
the momenta over P0 scale by P0_old / P0_new, dE = (E - E0) / (P0 c) keeps each particle's energy, and
z = s / beta0 - c t is kept, as Xsuite keeps zeta / beta0."""
function rereference!(r::AbstractMatrix, lost::AbstractVector, old::Beam, new::Beam)
    p_old = old.energy * old.beta
    p_new = new.energy * new.beta
    scale = p_old / p_new
    @inbounds for p in axes(r, 1)
        iszero(lost[p]) || continue
        r[p, 2] *= scale
        r[p, 4] *= scale
        r[p, 6] = (old.energy + r[p, 6] * p_old - new.energy) / p_new
    end
    return nothing
end

"""The bunch's coordinates, its losses, what it is tracked through, and the readings' arrays: the lattice
whole (a copy, its elements replaced as settings move) where TrackPad observes the bunch itself, else the
lattice cut into stretches at the observation points."""
function _setup(lat::Lattice, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector, rad::Integer=0)
    iszero(rad) || (lat = radiating(lat, rad))
    r = Matrix{Float64}(coords)
    lost = fill(false, size(r, 1))
    if OBSERVES
        through = Lattice(AbstractElement[lat.elements...]; name=lat.name, periodic=lat.periodic)
    else
        bounds = [0; Int.(cuts); length(lat.elements)]
        through = [Lattice(lat.elements[bounds[k] + 1:bounds[k + 1]]) for k in 1:length(bounds) - 1]
    end
    m = length(cuts)
    return r, lost, through, fill(NaN, m, turns), fill(NaN, m, turns), zeros(Int, m, turns)
end

"""Where each element of the lattice is in what the bunch is tracked through: nothing for the lattice
whole; (stretch, place in it) for the stretches."""
_places(::Lattice, cuts, n) = nothing
function _places(stretches::Vector, cuts, n)
    bounds = [0; Int.(cuts); n]
    place = Vector{Tuple{Int,Int}}(undef, n)
    for k in eachindex(stretches)
        for (j, i) in enumerate(bounds[k] + 1:bounds[k + 1])
            place[i] = (k, j)
        end
    end
    return place
end

"""Element `i` of the lattice replaced by `e` in what the bunch is tracked through."""
_place!(through::Lattice, place, i, e) = (through.elements[i] = e; nothing)
function _place!(stretches::Vector, place, i, e)
    k, l = place[i]
    stretches[k].elements[l] = e
    return nothing
end

"""Turns `span` of the call of the bunch `r`, the BPMs read after the elements `cuts`: through the lattice
whole in one `track!`, which reads them itself and spreads the bunch over Julia's threads (its random
numbers keyed by the ring's elements and the bunch's particles, so the threads do not change them); or
through the stretches, one after another (`_turn!`)."""
function _turns!(through::Lattice, r, b, lost, x, y, count, span::UnitRange{Int}, cuts, seed::Integer)
    m = length(cuts)
    observed = zeros(3, m, length(span))
    track!(r, through, b; nturns=length(span), turn=first(span), radiation_seed=seed, lost=lost,
           threaded=THREADED[] && Threads.nthreads() > 1 && size(r, 1) > 1,
           observe=Int.(cuts), observed=observed)
    @inbounds for (j, t) in enumerate(span), k in 1:m
        c = observed[1, k, j]
        count[k, t] = c
        if c > 0
            x[k, t] = observed[2, k, j] / c
            y[k, t] = observed[3, k, j] / c
        end
    end
    return nothing
end

function _turns!(stretches::Vector, r, b, lost, x, y, count, span::UnitRange{Int}, cuts, seed::Integer)
    for t in span
        _turn!(stretches, r, b, lost, x, y, count, t, length(cuts), seed)
    end
    return nothing
end

"""One turn through the segments, the centroid and particle count recorded after each of the first `m`."""
function _turn!(segments, r, b, lost, x, y, count, t, m, seed::Integer=0)
    n = size(r, 1)
    for (k, segment) in enumerate(segments)
        pass_segment!(r, segment, b, lost, t, k, seed)
        k > m && continue
        sx = 0.0
        sy = 0.0
        c = 0
        @inbounds for p in 1:n
            iszero(lost[p]) || continue
            sx += r[p, 1]
            sy += r[p, 3]
            c += 1
        end
        count[k, t] = c
        if c > 0
            x[k, t] = sx / c
            y[k, t] = sy / c
        end
    end
    return nothing
end

end # module
