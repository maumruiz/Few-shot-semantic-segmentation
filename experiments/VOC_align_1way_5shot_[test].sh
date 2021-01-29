python test.py with gpu_id=4 mode='test' snapshot='./runs/PANetExt_VOC_align_sets_0_1way_5shot_[train]/1/snapshots/30000.pth'
python test.py with gpu_id=4 mode='test' snapshot='./runs/PANetExt_VOC_align_sets_1_1way_5shot_[train]/1/snapshots/30000.pth'
python test.py with gpu_id=4 mode='test' snapshot='./runs/PANetExt_VOC_align_sets_2_1way_5shot_[train]/1/snapshots/30000.pth'
python test.py with gpu_id=4 mode='test' snapshot='./runs/PANetExt_VOC_align_sets_3_1way_5shot_[train]/1/snapshots/30000.pth'
python visualize.py with gpu_id=4 mode='visualize' model.align=True task.n_ways=1 task.n_shots=5
