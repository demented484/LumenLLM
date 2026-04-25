pub(crate) const BLACKWELL_FP4_KERNEL_SRC: &str = concat!(
    include_str!("kernels/blackwell/linear_quant.cu"),
    "\n",
    include_str!("kernels/blackwell/norm_rope_kv.cu"),
    "\n",
    include_str!("kernels/blackwell/attention.cu"),
    "\n",
    include_str!("kernels/blackwell/sampling.cu"),
);
