/opt/miniconda3/bin/micromamba run -n bidir python python/ray_rollout.py --checkpoint ./ray_results/base_distribute-007000-1009_215418 --config @data/rollout/angle.yaml --workers 1

ln -sf $CONDA_PREFIX/lib/libnvToolsExt.so.1 $CONDA_PREFIX/lib/libnvToolsExt.so
ln -s ${CONDA_PREFIX}/lib/libGL.so.1 ${CONDA_PREFIX}/lib/libGL.so
ln -s build/release/compile_commands.json .