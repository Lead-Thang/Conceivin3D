# Hunyuan3D-2/hy3dgen/shapegen/run_shape_gen.py

import argparse
import os
from pathlib import Path
from pipelines import Hunyuan3DDiTFlowMatchingPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='tencent/Hunyuan3D-DiT-v2-1')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--mesh_dir', type=str, required=True, help='Directory containing target GLB/OBJ files')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_name)

    # Add your dataset loader here
    image_paths = sorted([str(p) for p in Path(args.image_dir).glob('*.png')])
    mesh_paths = sorted([str(p) for p in Path(args.mesh_dir).glob('*.glb')])

    for epoch in range(args.num_epochs):
        for image_path, mesh_path in zip(image_paths, mesh_paths):
            print(f'Epoch {epoch} - Finetuning on {image_path}, {mesh_path}')
            mesh = pipeline(
                image=image_path,
                num_inference_steps=50,
                octree_resolution=380,
                num_chunks=20000,
                output_type='trimesh'
            )[0]
            mesh.export(f'{args.output_dir}/mesh_epoch{epoch}.glb')

if __name__ == "__main__":
    main()