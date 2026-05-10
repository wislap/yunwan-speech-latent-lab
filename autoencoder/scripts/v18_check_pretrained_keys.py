"""Dump which V18 phoneme encoder keys are NOT loaded from V17.3 teacher."""

import torch
from autoencoder.models.v18_encoder import V18PhonemeEncoder


def main():
    m = V18PhonemeEncoder()
    ckpt = torch.load(
        'outputs/models/v17_3_conformer_teacher_4spk_qc_1000/best.pt',
        map_location='cpu',
        weights_only=False,
    )
    state = ckpt.get('model', ckpt)

    # Simulate the load path's filter
    kept = {k for k in state if k.startswith(('mel.', 'conformer.'))}
    own = set(m.state_dict().keys())
    only_v18 = sorted(own - kept)

    print(f"V18 phoneme encoder total keys: {len(own)}")
    print(f"V17.3 candidate keys (mel/conformer): {len(kept)}")
    print(f"V18 keys NOT covered (random init):")
    for k in only_v18:
        print(f"  {k}  shape={list(m.state_dict()[k].shape)}")


if __name__ == "__main__":
    main()
