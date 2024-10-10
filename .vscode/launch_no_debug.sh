#!/bin/sh


/pox/pox.py samples.pretty_log smartController.tiger --wb_tracking=True --ai_debug=True --init_k_shot=3 \
    --batch_size=8 --node_features=False --wb_project_name=TIGER --wb_run_name=DDQN_pretrained_detached_backbone \
    --report_step_freq=50 --inference_freq_secs=1