reg_weights=(
    5e-4
    1e-3
)

# CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_laser_phi.sh "lr1e-2_step15_bs32_reg5e-4_init0.8__t1" "1e-2" "5e-4" "0.8"
for reg in "${reg_weights[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_laser_phi.sh "lr1e-2_step15_bs32_reg${reg}_init0.8__t" "1e-2" "${reg}" "0.8"
done