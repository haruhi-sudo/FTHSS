import torch
from typing import Optional, Dict, List, Tuple


def set_mask_flag(segment_dict: Dict[str, int], seg_type: str, 
                  target_id: str) -> Dict[str, Tuple[int, int]]:
    """
    Sets attention masks for each segment based on the given segment type and target ID.

    Args:
    - segment_dict: A dictionary of segment names and their lengths.
    - seg_type: One of ["P", "I", "O"], representing prefix, input, or output.
    - target_id: The segment ID to be targeted for causal attention.

    Returns:
    - A dictionary with segment IDs as keys and tuples (mask_type, length) as values.
    """
    assert seg_type in ["P", "I", "O"], "seg_type must be one of ['P', 'I', 'O']"
    assert target_id.startswith(seg_type), f"target_id should start with {seg_type}"

    mask_flag = {}

    # Handling for prefix (P)
    if seg_type == "P":
        for seg_id, length in segment_dict.items():
            if seg_id.startswith("P") and seg_id == target_id:
                mask_flag[seg_id] = (1, length)  # causal mask
            else:
                mask_flag[seg_id] = (0, length)  # zero mask

    # Handling for input (I)
    elif seg_type == "I":
        if target_id == "I_ALL":
            for seg_id, length in segment_dict.items():
                mask_flag[seg_id] = (1, length) if seg_id == target_id else (0, length)
        else:
            causal_or_full_or_zero = 2
            for seg_id, length in segment_dict.items():
                if seg_id.startswith("P") and seg_id[-1] == target_id[-1]: # I_A -> P_A, full mask for the corresponding prefix
                    mask_flag[seg_id] = (2, length)
                elif seg_id.startswith("P"): # zero mask for unrelated prefixes
                    mask_flag[seg_id] = (0, length)
                else:
                    if seg_id == target_id:
                        mask_flag[seg_id] = (1, length)
                        causal_or_full_or_zero = 0
                    else:
                        mask_flag[seg_id] = (causal_or_full_or_zero, length)
                
    # Handling for output (O)
    elif seg_type == "O":
        causal_or_full_or_zero = 2  # Initially assume full mask
        for seg_id, length in segment_dict.items():
            if seg_id.startswith("P"):
                if seg_id[-1] == target_id[-1]:
                    mask_flag[seg_id] = (2, length)  # full mask for the corresponding prefix
                else:
                    mask_flag[seg_id] = (0, length)  # zero mask for unrelated prefixes
            elif seg_id.startswith("I_ALL"): # full mask for I_ALL
                mask_flag[seg_id] = (2, length)  # full mask for input
            elif seg_id.startswith("I"): 
                if seg_id[-1] == target_id[-1] and causal_or_full_or_zero!=0: # full mask for the corresponding input
                    mask_flag[seg_id] = (2, length)
                else: # zero mask for other inputs
                    mask_flag[seg_id] = (0, length)
            elif seg_id.startswith("O"):
                if seg_id == target_id:
                    mask_flag[seg_id] = (1, length)  # causal mask for itself
                    causal_or_full_or_zero = 0  # Switch to zero mask for subsequent outputs
                else:
                    mask_flag[seg_id] = (causal_or_full_or_zero, length)  # full or zero mask for others

    return mask_flag

def mask_flag_to_mask(mask_flag: Dict[str, Tuple[int, int]], seq_len: int, 
                      dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Converts the mask flags to actual mask tensors.

    Args:
    - mask_flag: A dictionary of mask flags.
    - seq_len: The length of the input sequence.
    - dtype: The data type for the mask tensor.
    - device: The device to store the tensor on.

    Returns:
    - A tensor representing the combined mask for all segments.
    """
    mask = []
    for seg_id, (mask_type, other_seq_len) in mask_flag.items():
        if mask_type == 0:
            mask.append(_make_zero_mask((seq_len, other_seq_len), dtype, device))
        elif mask_type == 1:
            assert seq_len == other_seq_len, "seq_len should other_seq_len when mask_type is 1"
            mask.append(_make_causal_mask((seq_len, other_seq_len), dtype, device))
        elif mask_type == 2:
            mask.append(_make_full_mask((seq_len, other_seq_len), dtype, device))
    return torch.cat(mask, dim=-1)

def _make_full_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Creates a full mask tensor."""
    input_len, tgt_len = input_ids_shape
    mask = torch.full((input_len, tgt_len), 0, device=device)
    return mask.to(dtype)

def _make_zero_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Creates a zero mask tensor."""
    input_len, tgt_len = input_ids_shape
    mask = torch.full((input_len, tgt_len), torch.finfo(dtype).min, device=device)
    return mask.to(dtype)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0) -> torch.Tensor:
    """Creates a causal mask tensor."""
    _, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    return mask

def make_cascade_mask(segment_dicts: List[Dict[str, int]], dtype: torch.dtype,
                       device: torch.device, past_key_values_length: int=0) -> List[Dict[str, torch.Tensor]]:
    """
    Generates the cascade masks for a list of segment dictionaries.
    
    The segment dictionary splits the model's input-output sequence into different segments.
    Segments are categorized into prefix (P), input (I), and output (O).
    For example:
    >>> segment_dicts = [{
        "P_A": 50, "P_B": 50, "P_C": 50,  # learnable prefixes for different tasks
        "I_ALL": 100,  # model input
        "O_A": 16, "O_B": 16, "O_C": 16  # model output for different tasks
    }]

    IMPORTANT: segment_dict must be ordered!(A, B, C)

    Args:
    - segment_dicts: A list of dictionaries, each containing segment lengths.
    - dtype: The data type for the mask tensors.
    - device: The device to store the tensors on.

    Returns:
    - A list of dictionaries containing the mask tensors for each segment.
    """
    mask_dicts = []
    bsz = len(segment_dicts)  # batch size
    
    if past_key_values_length == 0:
        for segment_dict in segment_dicts:
            mask_dict = {}
            for seg_id, length in segment_dict.items():
                mask_flag = set_mask_flag(segment_dict, seg_type=seg_id[0], target_id=seg_id)
                mask_dict[seg_id] = mask_flag_to_mask(mask_flag, length, dtype, device)
            
            mask_dicts.append(mask_dict)
    else:
        for segment_dict in segment_dicts:
            # assert sum(segment_dict.values()) == past_key_values_length + 1
            total_length = 0
            mask_dict = {}
            for seg_id, length in segment_dict.items():
                total_length += length
                if total_length <= past_key_values_length:
                    continue
                mask_flag = set_mask_flag(segment_dict, seg_type=seg_id[0], target_id=seg_id)
                mask_dict[seg_id] = mask_flag_to_mask(mask_flag, length, dtype, device)

            last_key = list(segment_dict.keys())[-1]
            mask_flag[last_key] = (2, mask_flag[last_key][1])
            mask_dict[last_key] = mask_flag_to_mask(mask_flag, 1, dtype, device)
            
            mask_dicts.append(mask_dict)
    
    return mask_dicts

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


if __name__ == "__main__":
    segment_dicts = [{
        "P_A": 50, "P_B": 50, "P_C": 50,  # learnable prefixes for different tasks
        "I_ALL": 100,  # model input
        "I_A": 10,
        "O_A": 16, "O_B": 16, "O_C": 16  # model output for different tasks
    }]
    segment_dicts = [{"P_A": 50, "P_B": 50, 'I_ALL': 17, 'I_A': 675, 'O_A': 77, 'O_B': 5}]
    # Example usage:
    set_mask_flag(segment_dicts[0], seg_type="P", target_id="P_A")
    # make_cascade_mask(segment_dicts, dtype=torch.float32, device=torch.device("cuda:0"))
    # make_cascade_mask(segment_dicts, dtype=torch.float32, device=torch.device("cuda:0"), past_key_values_length=873)

