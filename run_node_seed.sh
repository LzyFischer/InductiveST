for model_name in STGODE
do
    for dataset_name in PEMS04 PEMS08 PEMS03
    do
        for node_seed in 1 2
        do
            for seed in 6 7 8
            do
                for wd in 0.0001 
                do
                    # Function to get GPU utilization for a given GPU ID
                    get_gpu_load() {
                        gpu_id=$1
                        load=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
                        echo "$load"
                    }

                    # Function to choose the GPU with the least load
                    choose_gpu_with_least_load() {
                        gpu_count=$(nvidia-smi --list-gpus | wc -l)
                        if [ $gpu_count -eq 0 ]; then
                            echo "No GPUs available."
                            exit 1
                        fi

                        # Initialize variables
                        min_load=100
                        chosen_gpu=""

                        # Loop through available GPUs
                        for ((gpu_id = 0; gpu_id < $gpu_count; gpu_id++)); do
                            load=$(get_gpu_load $gpu_id)
                            if [ -z "$load" ]; then
                                continue
                            fi

                            if ((load < min_load)); then
                                min_load=$load
                                chosen_gpu=$gpu_id
                            fi
                        done

                        echo "$chosen_gpu"
                    }

                    # Choose GPU with the least load
                    chosen_gpu=$(choose_gpu_with_least_load)

                    if [ -z "$chosen_gpu" ]; then
                        echo "No available GPUs or unable to determine GPU load."
                        # exit 1
                        chosen_gpu=3
                    fi

                    echo "Selected GPU: $chosen_gpu"

                    # Set the CUDA_VISIBLE_DEVICES environment variable to restrict execution to the chosen GPU
                    export CUDA_VISIBLE_DEVICES=$chosen_gpu


                    info="${model_name}_${dataset_name}_seed${seed}_nodeseed${node_seed}"

                    echo "Start ${info}"
                    output_file="log/${info}.log"

                    nohup python main.py \
                        --config "configs/${model_name}/${dataset_name}.yml" \
                        --seed $seed \
                        --wandb_name model_name dataset_name node_seed seed \
                        --node_seed $node_seed  > $output_file 2>&1 &
                    # pid=$!
                    sleep 10
                done
            done
        done
        pid=$!
        wait $pid
    done
done




