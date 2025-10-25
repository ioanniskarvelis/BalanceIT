import numpy as np

def calculate_empirical_class_distribution(df, class_label):
    # Count the frequency of each class
    class_counts = class_label.value_counts()

    # Calculate the total number of instances
    total_instances = len(df)

    # Calculate the empirical class distribution
    empirical_distribution = class_counts / total_instances

    return np.array(empirical_distribution)

def construct_iota_m(K, m):
    """
    Construct the distribution iota_m for a K-class problem with m minority classes.

    :param K: Total number of classes.
    :param m: Number of minority classes.
    :return: The distribution iota_m as a numpy array.
    """
    iota_m = np.zeros(K)

    # Case when there are no minority classes or all classes are minority
    if m == 0 or m == K:
        raise ValueError("m must be between 0 and K-1")

    # Set m minority classes to 0
    # Note: In practice, you might want to choose which classes are minority based on your dataset
    iota_m[:m] = 0

    # Set K-m-1 majority classes to 1/K
    iota_m[m:K-1] = 1 / K

    # Set one majority class with the remaining probability
    iota_m[K-1] = 1 - np.sum(iota_m)

    return iota_m

def chi_square_divergence(p, q):
    """Calculate Chi-square divergence between two distributions."""
    # Ensure the distributions are numpy arrays and have no zero elements
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q += 1e-10  # Avoid division by zero

    return np.sum((p - q) ** 2 / q)

def total_variation_distance(p, q):
    """Calculate Total Variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))

def kullback_leibler_divergence(p, q):
    """Calculate Kullback-Leibler Divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p += 1e-10
    q += 1e-10

    return np.sum(p * np.log(p / q))

def chebyshev_distance(p, q):
    """Calculate Chebyshev distance between two distributions."""
    return np.max(np.abs(p - q))

def euclidean_distance(p, q):
    """Calculate Euclidean distance between two distributions."""
    return np.sqrt(np.sum((p - q) ** 2))

def hellinger_distance(p, q):
    """Calculate Hellinger distance between two distributions."""
    return (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def calculate_imbalance_degree(zeta, e):
    """
    Calculate the imbalance degree using Euclidean distance.

    :param zeta: Empirical class distribution (numpy array).
    :param e: Balanced class distribution (numpy array).
    :return: Imbalance degree.
    """
    # Number of classes
    K = len(zeta)

    # Number of minority classes
    m = np.sum(zeta < (1.0 / K))
    # Calculate the Euclidean distance between zeta and e
    d_zeta_e_eu = euclidean_distance(zeta, e)
    # Calculate the Chebyshev distance between zeta and e
    d_zeta_e_ch = chebyshev_distance(zeta, e)
    # Calculate the Kullback-Leibler divergence between zeta and e
    d_zeta_e_kl = kullback_leibler_divergence(zeta, e)
    # Calculate the Hellinger distance between zeta and e
    d_zeta_e_hl = hellinger_distance(zeta, e)
    # Calculate the Total Variation distance between zeta and e
    d_zeta_e_tv = total_variation_distance(zeta, e)
    # Calculate the Chi-square divergence between zeta and e
    d_zeta_e_cs = chi_square_divergence(zeta, e)

    # Construct the distribution iota_m
    iota_m = construct_iota_m(K, m)

    # Calculate the Euclidean distance between iota_m and e
    d_iota_m_e_eu = euclidean_distance(iota_m, e)
    # Calculate the Chebyshev distance between iota_m and e
    d_iota_m_e_ch = chebyshev_distance(iota_m, e)
    # Calculate the Kullback-Leibler divergence between iota_m and e
    d_iota_m_e_kl = kullback_leibler_divergence(iota_m, e)
    # Calculate the Hellinger distance between iota_m and e
    d_iota_m_e_hl = hellinger_distance(iota_m, e)
    # Calculate the Total Variation distance between iota_m and e
    d_iota_m_e_tv = total_variation_distance(iota_m, e)
    # Calculate the Chi-square divergence between iota_m and e
    d_iota_m_e_cs = chi_square_divergence(iota_m, e)

    # Compute the imbalance ration
    ir = np.max(zeta) / np.min(zeta)

    # Compute the imbalance degree
    id_eu = d_zeta_e_eu / d_iota_m_e_eu + (m - 1)
    id_ch = d_zeta_e_ch / d_iota_m_e_ch + (m - 1)
    id_kl = d_zeta_e_kl / d_iota_m_e_kl + (m - 1)
    id_hl = d_zeta_e_hl / d_iota_m_e_hl + (m - 1)
    id_tv = d_zeta_e_tv / d_iota_m_e_tv + (m - 1)
    id_cs = d_zeta_e_cs / d_iota_m_e_cs + (m - 1)

    return ir,id_eu,id_ch,id_kl,id_hl,id_tv,id_cs

def plot_dataset(df, labels):
    results = {
        'metrics' : {},
        'plots' : {}
    }

    label_counts = labels.value_counts()
    gini_coefficient = 1 - sum((count / len(df)) ** 2 for count in label_counts)
    results['metrics']['Gini coefficient'] = gini_coefficient

    empirical_distribution = calculate_empirical_class_distribution(df, labels)
    # results['Empirical class distribution'] = empirical_distribution
    # results['Occurrences'] = label_counts

    K = len(empirical_distribution)
    e = np.ones(K) / K

    imbalance_degrees = calculate_imbalance_degree(empirical_distribution, e)
    results['metrics']['Imbalanced ratio'] = imbalance_degrees[0]
    results['metrics']['Imbalanced degree (EU)'] = imbalance_degrees[1]
    results['metrics']['Imbalanced degree (CH)'] = imbalance_degrees[2]
    results['metrics']['Imbalanced degree (KL)'] = imbalance_degrees[3]
    results['metrics']['Imbalanced degree (HL)'] = imbalance_degrees[4]
    results['metrics']['Imbalanced degree (TV)'] = imbalance_degrees[5]
    results['metrics']['Imbalanced degree (CS)'] = imbalance_degrees[6]

    results['plots']['series'] = label_counts.tolist()
    results['plots']['labels'] = label_counts.index.tolist()
    # plt.figure(figsize=(10, 6))
    # plt.pie(label_counts, labels=label_counts.index)
    # plt.title('Pie chart of label distribution')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.hist(labels, bins=np.arange(len(label_counts)+1)-0.5, edgecolor='black')
    # plt.title('Histogram of label distribution')
    # plt.xlabel('Label')
    # plt.ylabel('Count')
    # plt.xticks(rotation=90)
    # plt.show()

    return results