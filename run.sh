for i in ACE MAVEN
do
    for k in 5 10
    do
        for j in 1 2 3 4 42
        do
            if [ "$i" = "ACE" ]; then
                t=10
            else
                t=20
            fi

            python main.py \
                --data_root data/token_ids \
                --label_desciptions data/label_descriptions \
                --dataset $i \
                --seed $j \
                --shot_num $k \
                --class_num $t \
                --backbone bert-base-uncased \
                --no_freeze_bert \
                --lr 2e-5 \
                --decay 1e-4 \
                --step_size 1 \
                --gammalr 0.99 \
                --batch_size 4 \
                --device cuda:0 \
                --wandb \
                --project_name LEAF \
                --save_output output \
                --epochs 3 \
                --task_ep_time 6 \
                --use_weight_ce \
                --alpha_ce 0.3 \
                --aug_repeat_times 10 \
                --use_description \
                --num_description 3 \
                --ratio_loss_des_cl 0.1 \
                --uniform_ep 1 \
                --skip_eval_ep 0 \
                --patience 4 \
                --lora_rank 64 \
                --lora_alpha 64 \
                --lora_dropout 0.3 \
                --use_lora \
                --use_mole \
                --mole_num_experts 4 \
                --mole_top_k 2 \
                --use_general_expert \
                --entropy_weight 0.1 \
                --load_balance_weight 1 \
                --general_expert_weight 0.2 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1
        done
    done
done