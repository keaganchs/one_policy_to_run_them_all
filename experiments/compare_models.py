
import jax
import jax.numpy as jnp
import orbax.checkpoint
import os

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def inspect_checkpoint(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("Path does not exist.")
        return

    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    
    # Try to restore without args to get the structure
    # We might need to handle sharding if it fails like before
    try:
        # We use a dummy restore args with sharding if needed, but let's try simple restore first
        # If we don't provide item, it restores everything
        # But we might hit the sharding issue again.
        
        # Let's try to read metadata first
        metadata = checkpointer.metadata(path)
        print(f"Root metadata type: {type(metadata)}")
        print(f"Root metadata dir: {dir(metadata)}")
        
        # If it's StepMetadata, it might wrap the actual content
        # Try to access the content.
        # Based on previous output, it seemed to print a dictionary structure when we printed it directly.
        # But maybe that was __repr__.
        
        def count_from_metadata(meta_tree):
            count = 0
            try:
                # Traverse the dictionary
                if isinstance(meta_tree, dict):
                    for key, value in meta_tree.items():
                        count += count_from_metadata(value)
                elif isinstance(meta_tree, list):
                    for item in meta_tree:
                        count += count_from_metadata(item)
                elif hasattr(meta_tree, 'shape') and meta_tree.shape is not None:
                    # It's a leaf (ArrayMetadata or ScalarMetadata)
                    # Check if shape is iterable (it should be a tuple)
                    if isinstance(meta_tree.shape, (tuple, list)):
                        prod = 1
                        for dim in meta_tree.shape:
                            prod *= dim
                        count += prod
            except Exception:
                pass # Ignore errors for non-iterable things
            return count

        def get_params_count(component_metadata):
            if 'params' in component_metadata:
                # Usually params -> params
                p = component_metadata['params']
                if 'params' in p:
                    return count_from_metadata(p['params'])
                return count_from_metadata(p)
            return 0

        # Use item_metadata which contains the actual structure
        root_dict = metadata.item_metadata
        # print(f"Item metadata keys: {root_dict.keys()}")

        if 'policy' in root_dict:
            p_count = get_params_count(root_dict['policy'])
            print(f"Policy Params: {p_count}")
        
        if 'critic' in root_dict:
            c_count = get_params_count(root_dict['critic'])
            print(f"Critic Params: {c_count}")
            
        if 'policy' in root_dict and 'critic' in root_dict:
             print(f"Total Params: {p_count + c_count}")

        # Check for TRM layers in policy params
        has_trm = False
        def check_trm(meta_tree):
            if isinstance(meta_tree, dict):
                for key, value in meta_tree.items():
                    if "TRM" in key:
                        return True
                    if check_trm(value):
                        return True
            return False
            
        if 'policy' in root_dict and 'params' in root_dict['policy']:
             if check_trm(root_dict['policy']['params']):
                 print("TRM layers DETECTED in Policy.")
             else:
                 print("TRM layers NOT detected in Policy.")

        return # Stop here

        # But we don't know the structure to build the target.
        
        # However, orbax allows restoring with item=None, but it might require sharding in restore_args.
        # Let's try to construct a generic restore_args with sharding for everything.
        
        # Actually, if we just want to count parameters, maybe we can rely on metadata?
        # Metadata usually contains shapes.
        
        # Let's try to restore.
        restored = checkpointer.restore(path, item=None)
        
        if "policy" in restored:
            policy_params = restored["policy"]["params"]
            critic_params = restored["critic"]["params"]
            
            p_count = count_params(policy_params)
            c_count = count_params(critic_params)
            
            print(f"Policy Params: {p_count}")
            print(f"Critic Params: {c_count}")
            print(f"Total Params: {p_count + c_count}")
            
            # Print structure of policy params to see layers
            print("Policy Structure:")
            print(jax.tree_util.tree_map(lambda x: x.shape, policy_params))
            
        else:
            print("Could not find 'policy' in checkpoint.")
            print("Keys found:", restored.keys())

    except Exception as e:
        print(f"Error restoring checkpoint: {e}")
        # If it fails due to sharding, we might need to be more clever.
        # But let's see if it fails first.

if __name__ == "__main__":
    path_trm = "/home/holmes/projects/thesis/ebt/one_policy_to_run_them_all/experiments/runs/trm_talos/trm_talos_E0/1765542203/models/model_best_jax"
    path_ppo = "/home/holmes/projects/thesis/ebt/one_policy_to_run_them_all/experiments/runs/trm_talos/trm_talos_PPO_BASELINE_E0/1765552311/models/model_best_jax"
    
    print("--- TRM Model ---")
    inspect_checkpoint(path_trm)
    print("\n--- PPO Baseline Model ---")
    inspect_checkpoint(path_ppo)
