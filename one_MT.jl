using Random, Statistics, Distributions, Plots, Combinatorics, LinearAlgebra,CSV, DataFrames,StatsBase

task_id = parse(Int, ARGS[1])
const BIRTH_RATE = 1
const DEATH_RATE = 0.2
const BACKGROUND_GENE = 0
const PUSH_RATE = parse(Int, get(ARGS, 2, "1"))
const DIVER_N = parse(Float64, get(ARGS, 3, "0.001"))
const POSSION_MUTATION = parse(Int, get(ARGS, 4, "10"))
const FITNESS = 0.5

global seed_value = rand(UInt32)

Random.seed!(seed_value)

struct Cell
    parent_info::Union{Cell,Nothing}
    divide_id::UInt32
    generation::Int
    Cell(parent_info::Cell, divide_id::UInt32) = new(parent_info, divide_id,1+parent_info.generation)
    Cell(divide_id::UInt32) = new(nothing, divide_id,length(divide_id))
end



function count_frequencies(arr)
    freq = Dict{eltype(arr), Int}() 
    for x in arr
        if haskey(freq, x)
            freq[x] += 1
        else
            freq[x] = 1
        end
    end
    return freq
end

function find_nearest_empty_lattice_point(tissue, cell_x, cell_y, wx)
    empty_cell = [-1, -1]
    nearest_space_distance = Inf
    num_nearest_empty_cells = 0
    search_radius = wx

    for i in -search_radius:search_radius
        for j in -search_radius:search_radius
            dist = i^2 + j^2
            dist > search_radius^2 && continue
            
            x = cell_x + i  
            y = cell_y + j
            
            # check boundary
            if x < 1 || x > size(tissue,1) || y < 1 || y > size(tissue,2)
                continue
            end
            
            if tissue[x, y] == 0
                if dist < nearest_space_distance
                    nearest_space_distance = dist
                    empty_cell = [x, y]
                    num_nearest_empty_cells = 1
                elseif dist == nearest_space_distance
                    num_nearest_empty_cells += 1
                end
            end
        end
    end
    
    return empty_cell, nearest_space_distance, num_nearest_empty_cells
end

function choose_nearest_empty_lattice_point(tissue, cell_x, cell_y, search_radius, nearest_space_distance)
    nearest_empty_cells_x = []
    nearest_empty_cells_y = []
    
    for i in -search_radius:search_radius
        for j in -search_radius:search_radius
            dist = i^2 + j^2
            dist > search_radius^2 && continue
            
            x = cell_x + i
            y = cell_y + j
            # check boundary
            if x < 1 || x > size(tissue,1) || y < 1 || y > size(tissue,2)
                continue
            end
            
            if tissue[x, y] == 0 && dist == nearest_space_distance
                push!(nearest_empty_cells_x, x)
                push!(nearest_empty_cells_y, y)
            end
        end
    end
    
    rand_index = floor(Int, rand() * length(nearest_empty_cells_x)) + 1
    return nearest_empty_cells_x[rand_index], nearest_empty_cells_y[rand_index]
end

function move_cells_along_path(tissue, chain)
    reverse!(chain)
    for i in 1:length(chain)-1
        x, y = chain[i]
        tissue[x, y] = tissue[chain[i+1][1], chain[i+1][2]]
    end
end

function new_cell(tissue, chain, mut, pri, b_r, num_divi, mutated_id)
    reverse!(chain)
    tissue[chain[1][1], chain[1][2]] = num_divi

    if b_r == 1
        mut +=1
        push!(mutated_id, num_divi)
    
    elseif b_r >=1
        pri += 1
    end
    return mut, pri
end

function calculate_adjacent_sum(coords, prev_x0, prev_y0)
    results = []
    prev_x = prev_x0
    prev_y = prev_y0
    
    for (dx, dy) in coords
        prev_x += dx
        prev_y += dy
        push!(results, [prev_x, prev_y])
    end
    
    return results
end

function push_direction(cell_x, cell_y, empty_cell_x, empty_cell_y)
    r_mv =  empty_cell_x - cell_x 
    c_mv = empty_cell_y  - cell_y 
    path = []
    push_distance = 0.0
    
    # Assuming push_direct1, push_direct2, push_direct3, push_direct4 are predefined arrays
    push_direct1 = [[1,1],[1,0],[0,1]]
    push_direct2 = [[1,-1],[1,0],[0,-1]]
    push_direct3 = [[-1,1],[-1,0],[0,1]]
    push_direct4 = [[-1,-1],[-1,0],[0,-1]]
    # Initialize the path and other variables
    choose_push_list = []
    maximum_slope_move = min(abs(r_mv), abs(c_mv))
    times_slope_move = maximum_slope_move
    push_distance += sqrt(2) * times_slope_move  # Adjusting push_distance for diagonal moves
    
    if maximum_slope_move != 0
        if r_mv > 0 && c_mv > 0
            choose_push_list = push_direct1
        elseif r_mv > 0 && c_mv < 0
            choose_push_list = push_direct2
        elseif r_mv < 0 && c_mv > 0
            choose_push_list = push_direct3
        elseif r_mv < 0 && c_mv < 0
            choose_push_list = push_direct4
        end
    else
        if c_mv > 0
            choose_push_list = push_direct1
        elseif c_mv < 0
            choose_push_list = push_direct4
        elseif r_mv > 0
            choose_push_list = push_direct2
        elseif r_mv < 0
            choose_push_list = push_direct3
        end
    end
    
    # Append the appropriate moves to the path
    for i in 1:times_slope_move
        push!(path, choose_push_list[1])  # Append the first direction
    end
    
    for i in 1:(abs(r_mv) - times_slope_move)
        push!(path, choose_push_list[2])  # Append the second direction (horizontal)
        push_distance += 1
    end
    
    for i in 1:(abs(c_mv) - times_slope_move)
        push!(path, choose_push_list[3])  # Append the third direction (vertical)
        push_distance += 1
    end
    
    # Shuffle the path in place
    shuffle!(path)

    coords = calculate_adjacent_sum(path, cell_x, cell_y)
    return coords
end

function reaction(mut_pop, pri_pop)

    mut_bir_rate = (BIRTH_RATE + FITNESS) * mut_pop
    pri_bir_rate = BIRTH_RATE * pri_pop
    
    population_birth = [mut_bir_rate, pri_bir_rate]
    
    death_mut = DEATH_RATE * mut_pop
    death_pri = DEATH_RATE * pri_pop
    population_death = [death_mut, death_pri]
    
    total_rate = sum(population_birth) + sum(population_death)
    t = -log(rand()) / total_rate
    
    r = rand() * total_rate

    if r <= sum(population_birth)
        # birth
        cum_sum = 0.0
        for (i, rate) in enumerate(population_birth)
            cum_sum += rate
            if cum_sum < r
                continue
            else
                return :divide, i  # i=1 mut，i=2 primary
            end
        end
    else
        # death
        r_death = r - sum(population_birth)
        cum_sum = 0.0
        for (i, rate) in enumerate(population_death)
            cum_sum += rate
            if cum_sum < r_death
                continue
            else
                return :death, i  # # i=1 mut，i=2 primary
            end
        end
    end
end

function random_choose_cell(tissue, is_mutated, mutated_id)
    wx = size(tissue, 1)
    
    
    while true
        x = rand(1:wx)
        y = rand(1:wx)
        
        # tissue[x, y] != 0 && return x, y 
        is_mutated > 1 && !(tissue[x, y] in mutated_id) && tissue[x, y] != 0 && return x, y 
        is_mutated == 1 && tissue[x, y] in mutated_id && tissue[x, y] != 0 && return x, y

    end  
    
end  

function generate_poisson_values(n::Int, k::Real)
    rand(Poisson(k), n)
end

function update_wx(tissue, wx)
    new_wx = Int(floor(wx * 1.25))
    new_grid = zeros(Int, (new_wx, new_wx))
    offset = div(new_wx - wx, 2)
    new_grid[offset+1:offset+wx, offset+1:offset+wx] = tissue
    return new_grid, new_wx
end

function sample_square(cell_matrix, width, wx)
    x = rand(1:wx)
    y = rand(1:wx)
    while x < width || x > size(cell_matrix,1)-width || y < width || y > size(cell_matrix,2)-width
        x = rand(1:wx)
        y = rand(1:wx)
    end
    L = width
    half = L ÷ 2  # 5
    start_x = x - half + 1  # x - 4
    end_x = x + half        # x + 5
    start_y = y - half + 1  # y - 4
    end_y = y + half        # y + 5
    

    submatrix = cell_matrix[start_x:end_x, start_y:end_y]
    return submatrix,x,y
end

function get_matrix_info(cell_matrix,x,y,width,cell_id, mutant_set)
    L = width
    half = L ÷ 2  # 5
    start_x = x - half + 1  # x - 4
    end_x = x + half        # x + 5
    start_y = y - half + 1  # y - 4
    end_y = y + half        # y + 5
    all_divide_id = []
    
    sampling_cells_composition = 0 # 0 pure WT, 1 pure Mut, 0-1 Mix
    
    # 提取子矩阵
    submatrix = cell_matrix[start_x:end_x, start_y:end_y]
    cell_num = count(x->(x!=0),submatrix)

    for cell in submatrix
        if cell == 0
            continue
        end
        
        for id in divition_ids(cell_id[cell])
            push!(all_divide_id, id)
        end
    end
    check_if_mix = in.(submatrix, Ref(mutant_set))
    
    if sum(check_if_mix) == 0
        sampling_cells_composition = 0  # all WT
    elseif sum(check_if_mix) == cell_num
        sampling_cells_composition = 1  # all Mut
    else
        sampling_cells_composition = (sum(check_if_mix))/cell_num  # Mix
    end
    return all_divide_id, sampling_cells_composition
end

function divition_ids(cell::Cell)
    ids = Int32[]
    extractor = cell
    while !isnothing(extractor.parent_info)
        push!(ids, Int32(extractor.divide_id))
        extractor = extractor.parent_info
    end
    return ids
end

function label_mutation_burden(cell_matrix::AbstractMatrix{Int},cell_id,wx,possion_value)
    mb_matrix = zeros(Int, (wx, wx))
    for i in 1:size(cell_matrix, 1)
        for j in 1:size(cell_matrix, 2)
            if cell_matrix[i, j] != 0  # cell exist
                divide_id = divition_ids(cell_id[cell_matrix[i, j]])
                mb_matrix[i, j] = sum(possion_value[collect(divide_id)])
            end
        end
    end
    return mb_matrix
end

function label_cells!(cell_matrix::AbstractMatrix{Int}, mutant_set::Set{Any})
    for i in 1:size(cell_matrix, 1)
        for j in 1:size(cell_matrix, 2)
            if cell_matrix[i, j] != 0  
                cell_id = cell_matrix[i, j]   
            
                cell_matrix[i, j] = (cell_id in mutant_set) ? 2 : 1
            end
        end
    end
    return cell_matrix
end


function calculate_genetic_distance(cell_matrix,x,y,width,cell_id,possion_value, mutant_set)
    L = width
    half = L ÷ 2  # 5
    start_x = x - half + 1  # x - 4
    end_x = x + half        # x + 5
    start_y = y - half + 1  # y - 4
    end_y = y + half        # y + 5
    all_distance = 0
    
    submatrix = cell_matrix[start_x:end_x, start_y:end_y]
    sampling_cells_composition = 0 # 0 pure WT, 1 pure Mut, 0-1 Mix
    cell_num = count(x->(x!=0),submatrix)
    pair_num = 0
    for (cell1,cell2) in combinations(1:length(submatrix),2)
        if submatrix[cell1] == 0 || submatrix[cell2] == 0
            continue
        else
            genetic_id = []
            for id in divition_ids(cell_id[submatrix[cell1]])
                push!(genetic_id, id)
            end
            
            for id in divition_ids(cell_id[submatrix[cell2]])
                push!(genetic_id, id)
            end
            freqs = countmap(genetic_id)
            common_elements = filter(kv -> kv[2] == 2, freqs) |> keys |> collect 
	    pair_num +=1
	    if length(common_elements)>0
	    
            	unique_elements = filter(kv -> kv[2] == 1, freqs) |> keys |> collect
            	all_distance+=sum(possion_value[collect(unique_elements)])/sum(possion_value[collect(common_elements)]) #sum(possion_value[collect(unique_elements)]
	    end
        end
    end
    check_if_mix = in.(submatrix, Ref(mutant_set))

    if sum(check_if_mix) == 0
        sampling_cells_composition = 0  # all WT
    elseif sum(check_if_mix) == cell_num
        sampling_cells_composition = 1  # all Mut
    else
        sampling_cells_composition = (sum(check_if_mix))/cell_num  # Mix
    end
    return all_distance/pair_num, sampling_cells_composition
end


function euclidean_distance(p1, p2)
    return norm(p2 - p1)
end

function plot_scatter(cell_matrix,cell_id,wx,possion_value,mut,mutant_set)
    global seed_value
    fig = scatter()
    g_df = DataFrame(genetic_distance=Float64[], sampling_state=Float64[],d_c=Float64[])
    df = DataFrame(ji=Float64[], d=Float64[], s=Int[], s_i=Float64[], s_j=Float64[])
    for sampling_length in [10,30,50]#50,60]
    
        print(sampling_length)

        jaccard_index = Float32[]
        coordinates = []
        distance = Float32[]

        while length(coordinates) < 100
            sample,sample_x,sample_y = sample_square(cell_matrix,sampling_length,wx)
            while count(x->x>0,sample) < sampling_length*sampling_length*0.95
                sample,sample_x,sample_y = sample_square(cell_matrix,sampling_length,wx)
            end
            push!(coordinates,[sample_x,sample_y])
	    if sampling_length == 10
            	g_distance, sampling_statues = calculate_genetic_distance(cell_matrix,sample_x,sample_y,sampling_length,cell_id,possion_value, mutant_set)
            	push!(g_df, (g_distance,sampling_statues,euclidean_distance([sample_x,sample_y],[div(wx, 2) + 1,div(wx, 2) + 1])))
            
	    end            
            
        end        
        

        index = 1
        for (i, j) in combinations(1:length(coordinates), 2)
            
            if euclidean_distance(coordinates[i],coordinates[j]) < sampling_length*2
                continue
            else
                set_mutations_i,sampling_state_i = get_matrix_info(cell_matrix,coordinates[i][1],coordinates[i][2],sampling_length,cell_id, mutant_set)
            
                set_mutations_j,sampling_state_j = get_matrix_info(cell_matrix,coordinates[j][1],coordinates[j][2],sampling_length,cell_id, mutant_set)
                
                push!(distance, euclidean_distance(coordinates[i],coordinates[j]))
                intersect_mutation = intersect(Set(set_mutations_i),Set(set_mutations_j))
                union_mutation = union(Set(set_mutations_i),Set(set_mutations_j))
		num_interse_mutation = sum(possion_value[collect(intersect_mutation)]) # 0 JI suggest no common ancester
                num_union_mutation = sum(possion_value[collect(union_mutation)])
                push!(jaccard_index, num_interse_mutation/num_union_mutation)
                push!(df, (num_interse_mutation/num_union_mutation, euclidean_distance(coordinates[i],coordinates[j]), sampling_length,sampling_state_i,sampling_state_j))

	    end
            index+=1
        end
        # savefig(fig, "./scattertest_" * string(seed_value) * "_" * string(sampling_length) * ".pdf")
        # sfs=values(count_frequencies(mutations))

        fig=scatter!(distance,jaccard_index,label=sampling_length*sampling_length,xlabel="Distance", ylabel="Jaccard Index", title="P="*string(PUSH_RATE)*" f_mut="*string(mut/100000), ylims=(0, 0.5), size=(800, 600))
    savefig(fig, "/YOUR_PATH/" * string(PUSH_RATE) * "/" * string(POSSION_MUTATION) * "/scatter_" * string(seed_value)* "_" * string(mut/100000) * "_" * string(PUSH_RATE) * ".pdf")
    CSV.write("/YOUR_PATH/" * string(PUSH_RATE) * "/" * string(POSSION_MUTATION) * "/ji_sim_" * string(seed_value)* "_" * string(mut/100000) * "_" * string(PUSH_RATE) * ".csv", df)
    CSV.write("/YOUR_PATH/" * string(PUSH_RATE) * "/" * string(POSSION_MUTATION) * "/gd_sim_" * string(seed_value) * "_" *string(mut/100000)* "_" * string(PUSH_RATE) * ".csv", g_df)
    end
    
end

function main(end_population=10000)
    global seed_value
    # Initialize the grid and birth/death rates
    wx = 20
    cell_matrix = zeros(Int, (wx, wx))
    center = div(wx, 2) + 1  # Julia uses 1-based indexing
    cell_matrix[center, center] = 1 # cell id start from 1
    mutated_id = Set([])
    # Initialize dictionaries properly
    mut = 0
    primary = 1
    index = 0
    cell_x, cell_y = center, center
    initial_cell = Cell(UInt32(0))
    num_divi = 1
    cell_id = []
    push!(cell_id,initial_cell)
    check_if_mutated = 0
    # Main loop
    # Random.seed!(seed_value)
    while mut + primary < end_population
        # if mut == 0 && mut + primary>100
        #     return mut+primary
        # end
        index += 1
        if maximum(abs.([cell_x, cell_y] .- center)) > div(wx, 2) - 3
            cell_matrix, wx = update_wx(cell_matrix, wx)
            center = div(wx, 2) + 1
        
        end
        event, cell_b_r = reaction(mut, primary)
    
        cell_x, cell_y = random_choose_cell(cell_matrix, cell_b_r,mutated_id)
    
        if event == :divide
            
            empty_cell, nearest_dist, num_empty = find_nearest_empty_lattice_point(
                cell_matrix, cell_x, cell_y, PUSH_RATE)  # Make sure push is defined
            
            if empty_cell[1] == -1
                continue
            end
            
            empty_cell_x, empty_cell_y = empty_cell[1], empty_cell[2]
            if num_empty > 1
                empty_cell_x, empty_cell_y = choose_nearest_empty_lattice_point(cell_matrix, cell_x, cell_y, PUSH_RATE, nearest_dist)
            end
            
            coords = push_direction(cell_x, cell_y, empty_cell_x, empty_cell_y)
            move_cells_along_path(cell_matrix, coords)
            
            if (mut + primary)/end_population == DIVER_N && cell_b_r == 2 && check_if_mutated == 0
                cell_b_r = 1
                check_if_mutated = 1
                
            end

            

            if in(cell_matrix[cell_x, cell_y], mutated_id)
                
                push!(mutated_id, num_divi+1)
            end

            cell_1 = Cell(cell_id[cell_matrix[cell_x, cell_y]], UInt32(num_divi + 1))
            cell_2 = Cell(cell_id[cell_matrix[cell_x, cell_y]], UInt32(num_divi + 2))

            
            push!(cell_id,cell_1)
            push!(cell_id,cell_2)
            num_divi +=1

            
            
            cell_matrix[cell_x, cell_y] = num_divi
             
            num_divi +=1
            
            mut,primary = new_cell(cell_matrix, coords, mut, primary, cell_b_r, num_divi, mutated_id)
            
            
        else event == :death
            
            if cell_b_r < 2
                mut -=1
                
                delete!(mutated_id, cell_matrix[cell_x, cell_y])
            elseif cell_b_r ==2
                primary -=1
            end
            cell_matrix[cell_x,cell_y] = 0
        end
        
        if mut + primary ==0
            
            return -1,-1,-1,index
        end
    end
    if  mut/(mut+primary) >=0
        possion_divide = generate_poisson_values(num_divi,POSSION_MUTATION)
        plot_scatter(cell_matrix,cell_id,wx, possion_divide,mut,mutated_id)
	
	mb_matrix = label_mutation_burden(cell_matrix::AbstractMatrix{Int},cell_id,wx,possion_divide)
	cmap = cgrad([:white, :orange,:red], rev=false)  

	mb_fig=heatmap(mb_matrix,
    	color=cmap,
        title="F_mut="*string(mut/100000),
        framestyle=:none,
        size=(800, 800),
        colorbar=true)        
	savefig(mb_fig, "/YOUR_PATH/" * string(PUSH_RATE) * "/" * string(POSSION_MUTATION) * "/mb_heatmap_" * string(seed_value)* "_" * string(mut/100000) * "_" * string(PUSH_RATE) * ".pdf")

	wt_mut = label_cells!(cell_matrix,mutated_id)
	cmap = [:white, :blue, :red]
        hm_fig=heatmap(wt_mut,
        color = cmap,
        clims = (0, 2),           
        colorbar_ticks=[0, 1, 2],
        title="F_mut="*string(mut/100000),
        framestyle=:none,
	size=(800, 800),
        colorbar=false)
        savefig(hm_fig, "/YOUR_PATH/" * string(PUSH_RATE)* "/" * string(POSSION_MUTATION) * "/heatmap_" * string(seed_value)* "_" * string(mut/100000) * "_" * string(PUSH_RATE) * ".pdf")
    end
    return cell_matrix, primary, mut, cell_id,wx, possion_divide ,num_divi
end
@time main(100000)
