from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
from evaluate import evaluate_mae
import torch
import os
import numpy as np
import pandas as pd
from optimal import calculateBaryCenter
from prueba import fit_smpl_to_target
from dotenv import load_dotenv
import trimesh

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    measurer = MeasureBody("smpl")
    measurer2 = MeasureBody("smpl")

    data_path = os.getenv('DATA_PATH')
    general_path = os.getenv('3DV_PATH')
    df = pd.read_csv(os.path.join(data_path, 'df_final3.csv'))
    gender = df['Gender'].values
    generated_betas_path_base = os.path.join(general_path, 'generated_betas_exp1_euclidean_gender_pond')
    real_betas_path_base = os.path.join(general_path, 'real_betas_exp1_euclidean_gender_pond')
    fold_base_path = os.path.join(general_path, 'folds_exp1', 'folds.npy')
    modalities = ['fat', 'antro', 'muscle', 'full']
    modalities = ['full']
    test_folds = np.load(fold_base_path, allow_pickle=True)
    
    all_modalities_mae = []
    all_modalities_mae_direct = [] 
    detailed_results_direct = [] 
    detailed_results_fitted = [] 

    for modality in modalities:
        gen_files_in_modality = [f for f in os.listdir(generated_betas_path_base) if modality in f]
        real_files_in_modality = [f for f in os.listdir(real_betas_path_base) if modality in f]
        print(f"Generated files for modality {modality}: {gen_files_in_modality}")
        print(f"Real files for modality {modality}: {real_files_in_modality}")
        
        modality_mae = []
        modality_mae_direct = []

        for fold, gen_file, real_file in zip(test_folds, gen_files_in_modality, real_files_in_modality):
            print(f"Processing files: {gen_file} and {real_file}")

            generated_data = np.load(os.path.join(generated_betas_path_base, gen_file), allow_pickle=True)
            real_data = np.load(os.path.join(real_betas_path_base, real_file), allow_pickle=True)

            acc_mae = []
            acc_mae_direct = []

            for i in range(len(real_data)):
                real_beta_values = real_data[i][0]
                real_gender_value = real_data[i][1]
                neighbor_betas = generated_data[i][0]
                neighbor_genders = generated_data[i][1]
                neighbor_weights = generated_data[i][2]

                real_gender_str = "MALE" if real_gender_value == 1 else "FEMALE"

                neighbor_vertices_list = []
                generated_betas_list = []

                for neighbor_idx in range(len(neighbor_betas)):
                    neighbor_beta = neighbor_betas[neighbor_idx]
                    neighbor_gender_value = neighbor_genders[neighbor_idx]
                    neighbor_gender_str = "MALE" if neighbor_gender_value == 1 else "FEMALE"
                
                    neighbor_betas_tensor = torch.tensor(neighbor_beta, dtype=torch.float32).unsqueeze(0)
                    measurer.from_body_model(gender=neighbor_gender_str, shape=neighbor_betas_tensor)
                    neighbor_vertices = measurer.verts
                    trimesh.Trimesh(vertices=measurer.verts, faces=measurer.faces, process=False).show()
                    neighbor_vertices_list.append(neighbor_vertices)
                    generated_betas_list.append(neighbor_beta)

                neighbor_vertices_array = np.array(neighbor_vertices_list)
                weighted_vertices_direct = np.average(neighbor_vertices_array, axis=0, weights=neighbor_weights)

                num_vertices = neighbor_vertices_list[0].shape[0]
                split1 = num_vertices // 4
                split2 = 2 * num_vertices // 4
                split3 = 3 * num_vertices // 4

                index_blocks = [
                    list(range(0, split1)),
                    list(range(split1, split2)),
                    list(range(split2, split3)),
                    list(range(split3, num_vertices))
                ]

                barycenter_full = np.zeros((num_vertices, 3))

                colors = [
                    [255, 94, 0, 255],
                    [0, 154, 255, 255],    
                    [0, 200, 83, 255],     
                    [220, 38, 127, 255],   
                ]
                for nei in neighbor_vertices_list:
                    # scene = trimesh.Scene()
                    for block_indices, col in zip(index_blocks, colors):
                        geo = trimesh.points.PointCloud(nei[block_indices], process=False, colors=col)
                        # geo.show()
                    #     scene.add_geometry(geo)
                    # scene.show()
                    

                for (i_block, block_indices), col in zip(enumerate(index_blocks), colors):
                    print(f"⏱️ Processing block {i_block+1}/{len(index_blocks)}")
                    block_indices = np.array(block_indices)
                    distributions = [v[block_indices] for v in neighbor_vertices_list]
                    
                    bary_block = calculateBaryCenter(distributions, neighbor_weights, func='free_support_barycenter')
                    # gen = trimesh.points.PointCloud(bary_block, process=False, colors=col)
                    # gen.show()
                    # scene.add_geometry(gen)


                    for idx_i, idx in enumerate(block_indices):
                        barycenter_full[idx] = bary_block[idx_i]
                
                # scene.show()
                trimesh.points.PointCloud(barycenter_full, process=False).show()

                pre_betas = np.average(generated_betas_list, axis=0, weights=neighbor_weights)

                # Method 1
                measurer.from_body_model(gender=real_gender_str, shape=torch.zeros(1, 10))
                measurer.verts = weighted_vertices_direct
                verts_direct = measurer.verts
                measurer.measure(measurer.all_possible_measurements)
                measurer.label_measurements(STANDARD_LABELS)
                mea_gen_direct = measurer.measurements.copy()

                barycenter_tensor = torch.tensor(barycenter_full, dtype=torch.float32)
                # Method 2
                barycenter_squeezed = barycenter_tensor.squeeze()
                betas, vs = fit_smpl_to_target(barycenter_squeezed, gender=real_gender_str, num_iters=1, smpl_model=measurer)
                measurer.from_body_model(gender=real_gender_str, shape=torch.tensor(betas))
                mesh = trimesh.Trimesh(vertices=measurer.verts, faces=measurer.faces, process=False)
                mesh.visual.vertex_colors = [[255, 85, 85]] * len(mesh.vertices)  # Predicho
                mesh.show()

                barycenter_final = barycenter_squeezed.unsqueeze(0)
                measurer.from_verts(gender=real_gender_str, verts=torch.tensor(barycenter_full, dtype=torch.float32))
                measurer.measure(measurer.all_possible_measurements)
                measurer.label_measurements(STANDARD_LABELS)
                verts_fitted = measurer.verts
                mea_gen_fitted = measurer.measurements.copy()
                mesh = trimesh.Trimesh(vertices=measurer.verts, faces=measurer.faces, process=False)
                mesh.show()
                
                # Real
                real_betas_tensor = torch.tensor(real_beta_values, dtype=torch.float32).unsqueeze(0)
                measurer2.from_body_model(gender=real_gender_str, shape=real_betas_tensor)
                verts_real = measurer2.verts
                measurer2.measure(measurer2.all_possible_measurements)
                measurer2.label_measurements(STANDARD_LABELS)
                mea_real = measurer2.measurements
                mesh = trimesh.Trimesh(vertices=measurer2.verts, faces=measurer2.faces, process=False)
                mesh.visual.vertex_colors = [[85, 255, 255]] * len(mesh.vertices)
                mesh.show()
                # Evaluación
                mae_direct = evaluate_mae(mea_gen_direct, mea_real)
                mae_fitted = evaluate_mae(mea_gen_fitted, mea_real)

                acc_mae.append(mae_fitted)
                acc_mae_direct.append(mae_direct)
                modality_mae.append(mae_fitted)
                modality_mae_direct.append(mae_direct)
                
                detailed_results_direct.append({
                    'modality': modality,
                    'fold': len(detailed_results_direct) // len(real_data),
                    'sample': i,
                    'mae': mae_direct
                })
                detailed_results_fitted.append({
                    'modality': modality,
                    'fold': len(detailed_results_fitted) // len(real_data),
                    'sample': i,
                    'mae': mae_fitted
                })

                print(f"MAE Direct Method: {mae_direct}")
                print(f"MAE SMPL Fitting Method: {mae_fitted}")
                print(np.linalg.norm(verts_direct - verts_real, axis=1).mean())
                print(np.linalg.norm(verts_fitted - verts_real, axis=1).mean())

            if acc_mae:
                mean_mae = {key: np.mean([d[key] for d in acc_mae]) for key in acc_mae[0].keys()}
                mean_mae_direct = {key: np.mean([d[key] for d in acc_mae_direct]) for key in acc_mae_direct[0].keys()}
                print(f"Mean MAE for fold (Fitted): {mean_mae}")
                print(f"Mean MAE for fold (Direct): {mean_mae_direct}")

        if modality_mae:
            mean_modality_mae = {key: np.mean([d[key] for d in modality_mae]) for key in modality_mae[0].keys()}
            mean_modality_mae_direct = {key: np.mean([d[key] for d in modality_mae_direct]) for key in modality_mae_direct[0].keys()}
            print(f"Mean MAE for modality {modality} (Fitted):", mean_modality_mae)
            print(f"Mean MAE for modality {modality} (Direct):", mean_modality_mae_direct)
            all_modalities_mae.append(mean_modality_mae)
            all_modalities_mae_direct.append(mean_modality_mae_direct)
    
    print("\nResults per modality:")
    print("-" * 50)
    for modality, mae_fitted, mae_direct in zip(modalities, all_modalities_mae, all_modalities_mae_direct):
        print(f"\n MODALITY: {modality.upper()}")
        print(f"   Fitted Method: {mae_fitted}")
        print(f"   Direct Method: {mae_direct}")

    if detailed_results_fitted and detailed_results_fitted[0]['mae']:
        measurement_keys = list(detailed_results_fitted[0]['mae'].keys())

        print(f"\n Analysis for {len(measurement_keys)} body measurements:")
        print("-" * 60)
        
        for measure_key in measurement_keys:
            print(f"\n MEASUREMENT: {measure_key}")

            for modality in modalities:
                fitted_values = [
                    result['mae'][measure_key] for result in detailed_results_fitted 
                    if result['modality'] == modality and measure_key in result['mae']
                ]
                
                direct_values = [
                    result['mae'][measure_key] for result in detailed_results_direct 
                    if result['modality'] == modality and measure_key in result['mae']
                ]
                
                if fitted_values and direct_values:
                    fitted_mean = np.mean(fitted_values)
                    fitted_std = np.std(fitted_values)
                    fitted_cv = (fitted_std / fitted_mean) * 100 if fitted_mean != 0 else 0
                    
                    direct_mean = np.mean(direct_values)
                    direct_std = np.std(direct_values)
                    direct_cv = (direct_std / direct_mean) * 100 if direct_mean != 0 else 0
                    
                    print(f"   {modality.upper()}:")
                    print(f"      Fitted Method  → Mean: {fitted_mean:.4f}, Std: {fitted_std:.4f}, CV: {fitted_cv:.2f}%")
                    print(f"      Direct Method   → Mean: {direct_mean:.4f}, Std: {direct_std:.4f}, CV: {direct_cv:.2f}%")

if __name__ == "__main__":
    main()