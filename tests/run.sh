#!/bin/bash
# Please run this script from repo's main directory.

output_file="output.txt"
rm -f "$output_file"

comm_bin="bin/bench"
ivf_bin=$(echo $comm_bin)_faiss
alsh_bin=$(echo $comm_bin)_alsh
kmeans_bin=$(echo $comm_bin)_kmeans
quant_bin=$(echo $comm_bin)_quantization

ivf_args=(1 2 4 8 16 32 64 128 256 512 1024 2048) # nprobe
alsh_tables=(8 16 24 32) # number of hash tables
alsh_functions=(15 25 35 45) # number of hash functions in one hash table
alsh_r=(10 20) # r - hash function param
scaling=('0.85' '0.95') # scaling coefficient
aug=(0 1 2) # Neyshabur, Shrivastava or no augmentation
kmeans_layers=(2 3 4 5 6 7 8 9) # layers count
op_trees="10 20 30 40 50 60 70 80"
op_trees_arr=(10 20 30 40 50 60 70 80)
quant_subspace=(16 32 64 128) # subspace count
quant_centroid=(16 32 64) # centroid count

line_id=('R@1 = ' 'R@10 = ' 'R@100 = ' 'Search time' 'Intersection')

write_results() {
    for i in {0..4}; do results[i]=$(echo "$res" | grep -E "${line_id[i]}"); done
    if [[ "$1" == "kmeans" ]]; then
        #for i in {0..3}; do echo "${results[i]}"; done
        i=1
        for opened_trees in "${op_trees_arr[@]}"; do
            printf "%s\t%s\t%s\t%s\t%s\t%s\t" "$1" "$2" "$3" "$4" "$5" "$opened_trees" >> "$output_file"
            for j in {0..4}; do
                 printf "%s\t" $(echo "${results[j]}" | sed "${i}q;d" | awk '{print $NF}') >> "$output_file"
            done
            printf "\n" >> "$output_file"
            ((i++))
        done
    else
        printf "%s\t%s\t%s\t%s\t%s\t%s\t" "$1" "$2" "$3" "$4" "$5" "$6" >> "$output_file"
        for i in {0..5}; do printf "%s\t" $(echo "${results[i]}" | awk '{print $NF}' ) >> "$output_file" ; done
        printf "\n" >> "$output_file"
    fi
}

run_ivf="1"
run_alsh="1"
run_kmeans="1"
run_quant="1"

if [[ -v run_ivf ]]; then
    for a in "${ivf_args[@]}"; do
        echo "Running ivf $a"
        res="$($ivf_bin $a 2>&1)"
        write_results "ivf" "$a" "0" "0" "0" "0"
    done
fi

if [[ -v run_alsh ]]; then
    for a in "${alsh_tables[@]}"; do
        for b in "${alsh_functions[@]}"; do
            for c in "${alsh_r[@]}"; do
                for d in "${aug[@]}"; do
                    for e in "${scaling[@]}"; do
                        # dummy parameter, really needed for Shrivastava
                        if [[ "$d" != "1" ]]; then
                            e="-1"
                        fi
                        echo "Running alsh $a $b $c $d $e"
                        res="$($alsh_bin $a $b $c $d $e 2>&1)"
                        write_results "alsh" "$a" "$b" "$c" "$d" "$e"       
                        # without Shrivastava augmentation run once only
                        if [[ "$d" != "1" ]]; then
                            break
                        fi
                    done
                done
            done
        done
    done
fi

if [[ -v run_kmeans ]]; then
    for a in "${kmeans_layers[@]}"; do
        for c in "${aug[@]}"; do
            for d in "${scaling[@]}"; do
                if [[ "$c" != "1" ]]; then
                    d="-1"
                fi
                echo "Running kmeans $a $c $d $op_trees"
                res="$($kmeans_bin $a $c $d $op_trees 2>&1)"
                # there's no "$e" or "0" at the end of parameters
                # because the last one will be actual number of opened trees
                write_results "kmeans" "$a" "$c" "0" "$d"
                if [[ "$c" != "1" ]]; then
                    break
                fi
            done
        done
    done
fi

if [[ -v run_quant ]]; then
    for a in "${quant_subspace[@]}"; do
        for b in "${quant_centroid[@]}"; do
                echo "Running quant $a $b"
                res="$($quant_bin $a $b 2>&1)"
                write_results "quant" "$a" "$b" "0" "0" "0"
        done
    done
fi
