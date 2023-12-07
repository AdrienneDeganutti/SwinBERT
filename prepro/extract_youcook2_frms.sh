python ./prepro/extract_frames.py \
--video_root_dir /mnt/welles/scratch/adrienne/YouCook2/raw_videos/training_mp4 \
--save_dir ./datasets/YouCook2/extracted_frames \
--video_info_tsv ./datasets/YouCook2/training/amended.training.img.tsv \
--num_frames 32 \
# --debug


python ./prepro/create_image_frame_tsv.py \
--dataset YouCook2 \
--split training \
--image_size 256 \
--num_frames 32 \