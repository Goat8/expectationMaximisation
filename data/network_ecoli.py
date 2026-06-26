import pandas as pd

def load_dream5_ecoli(
    expression_path,
    gene_ids_path,
    tf_path
):
    
    # loading expression matrix
    # shape shoule be: 805 conditions x 4511 genes
    expr = pd.read_csv(
        expression_path,
        sep='\t',
        header=0
    )
    
    # need to take Transpose: we want genes x conditions (4511 x 805)
    X = expr.values.T.astype(float)
    gene_ids = list(expr.columns)
    
    # loading gene names
    gene_map = pd.read_csv(
        gene_ids_path,
        sep='\t',
        comment='#',
        header=None,
        names=['id', 'name']
    )
    id_to_name = dict(
        zip(gene_map['id'], gene_map['name'])
    )
    gene_names = [
        id_to_name.get(g, g) for g in gene_ids
    ]
    
    # loading transcription factors
    tfs = pd.read_csv(
        tf_path,
        sep='\t',
        header=None
    )[0].tolist()
    
    # boolean mask: which genes are TFs
    is_tf = [g in tfs for g in gene_ids]
    
    print(f"Loaded DREAM5 E. coli data:")
    print(f"  Genes: {X.shape[0]}")
    print(f"  Conditions: {X.shape[1]}")
    print(f"  Transcription factors: {sum(is_tf)}")
    print(f"  Mean expression: {X.mean():.3f}")
    print(f"  Std expression: {X.std():.3f}")
    
    return X, gene_names, gene_ids, is_tf


def load_gold_standard(gold_standard_path):
    gs = pd.read_csv(
        gold_standard_path,
        sep='\t',
        header=None,
        names=['tf', 'target', 'label']
    )
    # Keep only positive edges (label == 1)
    gs = gs[gs['label'] == 1]
    print(f"Gold standard: {len(gs)} known regulatory edges")
    return gs