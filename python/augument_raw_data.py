# This code is code for converting the raw motion data to the refined training dataset for Bidirectional GaitNet


import argparse
from pysim import RayEnvManager
import numpy as np
import os
import sys
import csv

parser = argparse.ArgumentParser()

# Raw Motion Path
parser.add_argument("--motion", type=str, default="/home/gait/BidirectionalGaitNet_Data/UniformSampling/3rd_rollout/abs")  # raw motion path
parser.add_argument("--save", type=str, default="/home/gait/BidirectionalGaitNet_Data/UniformSampling/3rd_rollout/gaitdata")  # raw motion path
parser.add_argument("--env", type=str, default="/home/gait/BidirectionalGaitNet_Data/GridSampling/3rd_rollout/env.xml")
parser.add_argument("--name", type=str, default="refined_data")
parser.add_argument("--min_length", type=int, default=140)
parser.add_argument("--resolution", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--progress_every", type=int, default=50)
parser.add_argument("--log", type=str, default="")

# convert raw motion to refined motion (2cycles, 60 frames)
def convertToRefinedMotion(f, num_known_param, resolution=30, min_length=140, stats=None, errors=None):

    # refined_motion : [motion, knownparam, truthparam]
    # return : list of refined_motion
    result = []

    # make reference phi
    phis = []
    for i in range(resolution * 2):
        phis.append(i / resolution)

    # input_motions = [[],[]] # gait + knownparam , truth param

    try:
        loaded_file = np.load(f)
    except Exception as e:
        if errors is not None:
            errors.append({"file": f, "index": -1, "reason": "np_load_failed", "detail": str(e)})
        if stats is not None:
            stats["file_load_errors"] = stats.get("file_load_errors", 0) + 1
        return result

    loaded_params = loaded_file["params"]
    loaded_motions = loaded_file["motions"]
    loaded_lengths = loaded_file["lengths"]

    #     print("\t loaded params : ", len(loaded_param))
    i = 0
    for loaded_idx in range(len(loaded_lengths)):
        prev_phi = None
        param = loaded_params[loaded_idx]

        if stats is not None:
            stats["sequences_total"] = stats.get("sequences_total", 0) + 1

        if loaded_lengths[loaded_idx] > min_length:
            # Motion preprocessing for motion phi
            prev_phi = -1
            phi_offset = -1
            for j in range(loaded_lengths[loaded_idx]):
                if prev_phi > loaded_motions[i + j][-1]:
                    phi_offset += 1
                prev_phi = loaded_motions[i+j][-1]
                loaded_motions[i+j][-1] += phi_offset

            phi_idx = 0
            motion_idx = 0
            refined_motion = [[], param]
            while phi_idx < len(phis) and motion_idx < loaded_lengths[loaded_idx] - 1:
                if loaded_motions[i+motion_idx][-1] <= phis[phi_idx] and phis[phi_idx] < loaded_motions[i+motion_idx+1][-1]:
                    w1 = loaded_motions[i+motion_idx+1][-1] - phis[phi_idx]
                    w2 = phis[phi_idx] - loaded_motions[i+motion_idx][-1]
                    # Interpolate six dof pos
                    v1 = loaded_motions[i+motion_idx][3:6] - loaded_motions[i+motion_idx-1][3:6]
                    v2 = loaded_motions[i+motion_idx+1][3:6] - loaded_motions[i+motion_idx][3:6]

                    p = (w1 * env.posToSixDof(loaded_motions[i+motion_idx][:-1]) + w2 * env.posToSixDof(loaded_motions[i+motion_idx+1][:-1])) / (w1 + w2)
                    v = (w1 * v1 + w2 * v2) / (w1 + w2)

                    p[6] = v[0]
                    p[8] = v[2]

                    # Store pure motion data only (p) - parameters are handled separately
                    refined_motion[0].append(p)
                    phi_idx += 1
                else:
                    motion_idx += 1
            

            if len(refined_motion[0]) == 60:
                result.append(refined_motion)
                if stats is not None:
                    stats["segments_generated"] = stats.get("segments_generated", 0) + 1
            else:
                if stats is not None:
                    stats["bad_refined_length"] = stats.get("bad_refined_length", 0) + 1
                if errors is not None:
                    errors.append({"file": f, "index": int(loaded_idx), "reason": "refined_not_60", "detail": int(len(refined_motion[0]))})

        i += loaded_lengths[loaded_idx]
        if loaded_lengths[loaded_idx] <= min_length:
            if stats is not None:
                stats["sequences_too_short"] = stats.get("sequences_too_short", 0) + 1
            if errors is not None:
                errors.append({"file": f, "index": int(loaded_idx), "reason": "sequence_too_short", "detail": int(loaded_lengths[loaded_idx])})
    return result

def save_motions(motions, params):
    np.savez_compressed("new_motion", motions=motions, params=params)


if __name__ == "__main__":
    args = parser.parse_args()

    ## print args information 
    print("motion path : ", args.motion)
    print("save path : ", args.save)
    print("min_length : ", args.min_length, " resolution : ", args.resolution, " batch_size : ", args.batch_size)


    # Ensure save directory exists
    try:
        os.makedirs(args.save, exist_ok=True)
    except Exception as e:
        print("Failed to create save directory:", args.save, e)
        sys.exit(1)

    train_filenames = os.listdir(args.motion)
    train_filenames.sort()

    file_idx = 0

    # Environment Loeading
    env = RayEnvManager(args.env)

    # Loading all motion from file Data PreProcessing
    file_idx = 0
    save_idx = 0
    results = []
    stats = {
        "files_total": len(train_filenames),
        "npz_files": 0,
        "file_load_errors": 0,
        "sequences_total": 0,
        "sequences_too_short": 0,
        "segments_generated": 0,
        "bad_refined_length": 0,
    }
    errors = []

    print(len(train_filenames), ' files are loaded ....... ')
    # while True:
    while file_idx < len(train_filenames):
        f = train_filenames[file_idx % len(train_filenames)]
        file_idx += 1
        if f[-4:] != ".npz":
            # print(path, ' is not npz file')
            continue
        path = args.motion + '/' + f

        stats["npz_files"] += 1

        new_segments = convertToRefinedMotion(path, env.getNumKnownParam(), resolution=args.resolution, min_length=args.min_length, stats=stats, errors=errors)
        results += new_segments

        # Periodic progress
        if args.progress_every > 0 and (file_idx % args.progress_every == 0 or file_idx == len(train_filenames)):
            print(f"[{file_idx}/{len(train_filenames)}] npz:{stats['npz_files']} segs:{stats['segments_generated']} saved:{save_idx} too_short:{stats['sequences_too_short']} bad_len:{stats['bad_refined_length']} errors:{len(errors)}")

        if len(results) >= args.batch_size:
            res = results[:args.batch_size]
            motions = np.array([r[0] for r in res])
            params = np.array([r[1] for r in res])
            np.savez_compressed(args.save + "/" + args.name + "_" + str(save_idx), motions=motions, params=params)
            results = results[args.batch_size:]
            save_idx += 1

    # Save remaining results at end
    if len(results) > 0:
        motions = np.array([r[0] for r in results])
        params = np.array([r[1] for r in results])
        np.savez_compressed(args.save + "/" + args.name + "_" + str(save_idx), motions=motions, params=params)
        save_idx += 1

    # Optional error log
    if args.log:
        try:
            with open(args.log, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["file", "index", "reason", "detail"])
                writer.writeheader()
                for e in errors:
                    writer.writerow(e)
            print("Wrote error log:", args.log)
        except Exception as e:
            print("Failed to write error log:", e)

    # Final summary
    print("Processing complete.")
    print(f"Files total: {stats['files_total']} (npz: {stats['npz_files']})")
    print(f"Sequences total: {stats['sequences_total']} | too_short: {stats['sequences_too_short']}")
    print(f"Segments generated: {stats['segments_generated']} | bad_len: {stats['bad_refined_length']}")
    print(f"Saved files: {save_idx} | errors: {len(errors)} | file_load_errors: {stats['file_load_errors']}")
