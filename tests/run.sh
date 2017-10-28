#!/bin/bash
# Please run this script from repo's main directory.

rm -f output.txt

comm_bin="bin/bench"
time_path="/usr/bin/time -v"
ivf_bin=$(echo $comm_bin)_faiss
alsh_bin=$(echo $comm_bin)_alsh
kmeans_bin=$(echo $comm_bin)_kmeans
quant_bin=$(echo $comm_bin)_quantization

# TODO decide which arguments we'll test
ivf_args=(1 2 4 8 16 32 64 128) #16 32 64 128 256 512 1024) # nprobe
alsh_args_1=(2 3 4 6 8 12 16 24 32) # number of hash tables
alsh_args_2=(2 3 4 6 8 12 16 24 32) # number of hash function in one hash table
kmeans_args_1=(3) # number of additional components (m)
kmeans_args_2=(2 3 4 5) # layers count
kmeans_args_3=(1 2 3 4 5) # opened trees
quant_args_1=(16 32 48 64 128) # subspace count
quant_args_2=(16 32 64) # centroid count

line_id=('R@1 = ' 'R@10 = ' 'R@100 = ' 'Train time' 'Add time' 'Search time' 'Maximum resident')

write_results() {
    for i in {0..6}; do results[i]=$(echo "$res" | grep -E "${line_id[i]}" | awk '{ print $NF }'); done
    printf "%s" "$1" >> output.txt
    for i in {0..6}; do printf " %s" "${results[i]}" >> output.txt ; done
    printf "\n" >> output.txt
}

for a in "${ivf_args[@]}"; do
    res="$($time_path $ivf_bin $a 2>&1)"
    write_results "ivf $a 0 0"
done

#for a in "${alsh_args_1[@]}"; do
#    for b in "${alsh_args_2[@]}"; do
#        res="$($time_path $alsh_bin $a $b 2>&1)"
#        write_results "alsh $a $b 0"
#    done
#done

for a in "${kmeans_args_1[@]}"; do
    for b in "${kmeans_args_2[@]}"; do
        for c in "${kmeans_args_3[@]}"; do
            res="$($time_path $kmeans_bin $a $b $c 2>&1)"
            write_results "kmeans $a $b $c"
        done
    done
done

for a in "${quant_args_1[@]}"; do
    for b in "${quant_args_2[@]}"; do
            res="$($time_path $quant_bin $a $b 2>&1)"
            write_results "quant $a $b 0"
    done
done
