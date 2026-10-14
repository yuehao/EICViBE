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

function element(kind::Symbol, name::Symbol, r::Row, beam::Beam)
    if kind === :drift
        return Drift(r[:L]; name=name)
    elseif kind === :marker
        return Marker(; name=name)
    elseif kind === :quadrupole
        return Quadrupole(r[:L], r[:b1]; name=name, polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                          max_order=order(r), num_int_steps=steps(r), placement(r)...)
    elseif kind === :sextupole
        return Sextupole(r[:L], 2 * r[:b2]; name=name, polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                         max_order=order(r), num_int_steps=steps(r), placement(r)...)
    elseif kind === :octupole
        return Octupole(r[:L], 6 * r[:b3]; name=name, polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                        max_order=order(r), num_int_steps=steps(r), placement(r)...)
    elseif kind === :bend
        return SBend(r[:L], r[:angle], r[:e1], r[:e2]; name=name,
                     fint1=r[:fint1], fint2=r[:fint2], gap=r[:gap],
                     polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                     max_order=order(r), num_int_steps=steps(r), placement(r)...)
    elseif kind === :exactbend
        return ExactSBend(r[:L], r[:angle], r[:e1], r[:e2]; name=name, gk=r[:gk],
                          polynom_a=polynom_a(r), polynom_b=polynom_b(r),
                          max_order=order(r), num_int_steps=steps(r), placement(r)...)
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

"""Coordinates (6 x elements+1) at the entrance and after every element, from `start`."""
function along(lat::Lattice, b::Beam, start::AbstractVector)
    r = SVector{6,Float64}(start...)
    out = Matrix{Float64}(undef, 6, length(lat.elements) + 1)
    out[:, 1] = r
    inverse_beta = 1 / b.beta
    for (i, e) in enumerate(lat.elements)
        r = TrackPad.pass!(e, r, inverse_beta)
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
chromaticity are exact derivatives, with no finite-difference steps. With
`inside`, magnets are sampled at their integrator steps (one piece per step,
each integrated in one step), which adds points inside them and changes no
result.
"""
function optics(lat::Lattice, b::Beam, start, reference::AbstractVector, inside::Bool=false)
    ref = SVector{6,Float64}(reference...)
    sampling = (sample_integrator_steps=inside,)
    tw = start === nothing ? twiss(lat, b; reference=ref, method=:tpsa, sampling...) :
         twiss(lat, b; entrance=start, reference=ref, method=:tpsa, sampling...)
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
function observe(lat::Lattice, b::Beam, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector)
    r = Matrix{Float64}(coords)
    n = size(r, 1)
    lost = zeros(Int, n)
    bounds = [0; Int.(cuts); length(lat.elements)]
    segments = [Lattice(lat.elements[bounds[k] + 1:bounds[k + 1]]) for k in 1:length(bounds) - 1]
    record = fill(NaN, length(cuts), turns, n, 4)
    for t in 1:turns
        for (k, segment) in enumerate(segments)
            isempty(segment.elements) || linepass!(r, segment, b, lost)
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
function centroids(lat::Lattice, b::Beam, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector)
    r, lost, segments, x, y, count = _setup(lat, coords, turns, cuts)
    for t in 1:turns
        _turn!(segments, r, b, lost, x, y, count, t, length(cuts))
    end
    return x, y, count, r, lost
end

"""
    centroids(lat, b, coords, turns, cuts, first, count, kinds, names, columns, base, group, ramp, delta, fraction)

As `centroids`, while settings move turn by turn. Group `g` is the pieces `first[g]` to
`first[g] + count[g] - 1` of `lat` (1-based), one occurrence; its rows are the next `count[g]` rows of
`base` (and of `kinds`, `names`). A contribution `c` adds `fraction[ramp[c], t] * delta_c` to group
`group[c]`'s rows, `delta_c` being its next `count[group[c]]` rows of `delta`. Before each turn whose
fractions changed, the group's elements are built anew from those rows: element inputs are linear in
the settings, so a magnet follows its supply's ramp exactly, turn by turn, and nothing steps.
"""
function centroids(lat::Lattice, b::Beam, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector,
                   first::AbstractVector, count::AbstractVector, kinds::AbstractVector, names::AbstractString,
                   columns::AbstractString, base::AbstractMatrix, group::AbstractVector, ramp::AbstractVector,
                   delta::AbstractMatrix, fraction::AbstractMatrix)
    r, lost, segments, x, y, counts = _setup(lat, coords, turns, cuts)
    bounds = [0; Int.(cuts); length(lat.elements)]
    place = Vector{Tuple{Int,Int}}(undef, length(lat.elements))
    for k in eachindex(segments)
        for (j, i) in enumerate(bounds[k] + 1:bounds[k + 1])
            place[i] = (k, j)
        end
    end
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
    for t in 1:turns
        for g in 1:groups
            if t > 1 && all(fraction[Int(ramp[c]), t] == fraction[Int(ramp[c]), t - 1] for c in contributions[g])
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
                            Row(view(part, j, :), index), b)
                k, l = place[Int(first[g]) + j - 1]
                segments[k].elements[l] = e
            end
        end
        _turn!(segments, r, b, lost, x, y, counts, t, length(cuts))
    end
    return x, y, counts, r, lost
end

"""The bunch's coordinates, its losses, the lattice cut at the observation points, and the readings' arrays."""
function _setup(lat::Lattice, coords::AbstractMatrix, turns::Integer, cuts::AbstractVector)
    r = Matrix{Float64}(coords)
    lost = zeros(Int, size(r, 1))
    bounds = [0; Int.(cuts); length(lat.elements)]
    segments = [Lattice(lat.elements[bounds[k] + 1:bounds[k + 1]]) for k in 1:length(bounds) - 1]
    m = length(cuts)
    return r, lost, segments, fill(NaN, m, turns), fill(NaN, m, turns), zeros(Int, m, turns)
end

"""One turn through the segments, the centroid and particle count recorded after each of the first `m`."""
function _turn!(segments, r, b, lost, x, y, count, t, m)
    n = size(r, 1)
    for (k, segment) in enumerate(segments)
        isempty(segment.elements) || linepass!(r, segment, b, lost)
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
