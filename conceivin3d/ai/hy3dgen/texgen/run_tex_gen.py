# Hunyuan3D-2/hy3dgen/texgen/run_tex_gen.py

import argparse
from pipelines import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

def main():
    parser = argparse.ArgumentParser(description="Apply texture to 3D mesh")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to input mesh")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # Load pipeline
    config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    pipeline = Hunyuan3DPaintPipeline(config)
    result = pipeline(mesh_path=args.mesh_path, image_path=args.image_path)

    print("Textured mesh:", result)

if __name__ == "__main__":
    main()