VARS_DESCRIPTION = """
  Vars:             Description:                                 Value in papers
                                                               Vaswani     Music     
  -----             -------------                                ---------------
INPUT/OUTPUTS:
    X               input
    max_seq         length of the longest sequence allowed        ?         3500
                    ** Only needed for usual embedding.               
    Y               input of the decoder

GENERAL
    nheads          nb heads. Could be different in each layer and       8 
                    variables have been defined, but we will use the
                    same value for all
    d_model         embedding size. Size of all layers needs             512
                    to be constant. =D in the music paper.
                                    
* Note. Dropbout rate will be the same everywhere
     
EMBEDDING
    dropout         dropout rate                                        0.1
        
ENCODER
    n_layers_e      nb of encoding layers                                6
    dropout         dropout rate                                        0.1
    
ENCODING LAYERS
    h_e = nheads    nb heads                                             8
    W1, W2          weights for the 2-layer NN of sub-layer 2
    b1, b2          biases for the 2-layer NN of sub-layer 2   

DECODER   
    n_layers_d      nb of decoding layers                                6
    dropout         dropout rate                                        0.1

DECODING LAYERS
    h_d1 = nheads   nb heads                                             8
    h_d2 = nheads   nb heads                                             8
    W1, W2          weights for the 2-layer NN of sub-layer 3
    b1, b2          biases for the 2-layer NN of sub-layer 3       

SCALED DOT PRODUCT ATTENTION of head i
    Q_i, K_i, V_i   a set of queries q, keys k, values v
    d_Q             dimension of a set of queries                   d_emb/h = 64
    d_K             dimension of a set of keys                           idem
    d_V             dimension of a set of values                         idem

MULTI-HEAD ATTENTION
    Q, K, V         Final learned queries, keys and values
    W_Q             |  Projection matrices                dim = (h, d_emb, d_Q)  
    W_K             |  sending Q, K and V to              dim = (h, d_emb, d_K)     
    W_V             |  different subspaces for each head  dim = (h, d_emb, d_V)  
    W_O             Projection matrix for the final       dim = (h*d_emb, d_emb) 
                    attention output Z
"""

MODEL_DESCRIPTION = """

EMBEDDING: on X for the encoder, on Y for the decoder                                             
- Learned embedding
  + Add positional embedding
  + *sqrt(d_emb)
  + dropout.
        
ENCODING LAYERS
- sub-layer 1: multi-head attention
               + dropout
               + residual connection
               + normalization
    input = embedding or output from preceeding encoding layer
- sub-layer 2: 2-layer feed-forward network 
               = linear (W1, b1) + ReLu + linear (W2, b2) 
               + dropout
               + residual connection
               + normalization
    input = output from sub-layer 1, each position separately 

DECODING LAYERS
- sub-layer 1: SAME
    input = embedding of the real Y or ouput from preceeding decoding layer
- sub-layer 2: SAME as in encoding but also received input from encoder but
               attention is masked to past
    input = output form sub-layer 1 + output from last encoding layer
- sub-layer 3: SAME as layer 2 in encoding + attention is masked to past
    input = output from sub-layer 2

SCALED DOT PRODUCT ATTENTION
- Could learn q_i, k_i, v_i = query, key, value for one input, one head i
    of dimensions d_q, d_k, d_v                                    
- It is faster to learn Q_i, K_i, V_i = a set of q, v, k together 
    (for one head i).

MULTI-HEAD ATTENTION
- Will learn the final Q, K, V for all heads. 
- Needs to learn W_Q, W_K, W_V, projection matrices who will learn to separate
    Q, K, V into different heads. That's the use of multi-head. Learns to find
    information from different representation subspaces (different projections)
    of the archives. They also reduce dimensionality for each head.
- Concatenates all heads' outputs (Z_i) = Z
- Final projection of the results with matrix W_O. 
    
Notes.
- d_q and d_k should always be equal. d_v could be different but is equal in our
    chosen model.
- To help with dimensions with residual connections, output of each sub-layer 
    is always d_emb.

"""