
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


def running_max_stats(scores):
    maxscores = np.maximum.accumulate(scores, axis=1)
    mean = np.mean(maxscores, axis=0) 
    std = np.std(maxscores, axis=0)
    return (maxscores, mean, std)

def plot_searcher_comparison():
    folder_path = 'logs/searcher_comparison'
    searchers = ['rand', 'mcts', 'mcts_bi', 'smbo']
    # searchers = ['smbo']
    search_space_type = 'deepconv'
    num_repeats = 5

    # loading the scores`
    m_maxscores = []
    std_maxscores = []
    for searcher_type in searchers:
        # load all the pickles
        rs = []
        for i in xrange(num_repeats):
            file_name = '%s_%s_%d.pkl' % (search_space_type, searcher_type, i)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                r = pickle.load(f)
            rs.append(r)

        srch_scores = [r['scores'] for r in rs]
        # from pprint import pprint
        # print searcher_type, len(srch_scores[0])
        #pprint(srch_scores)
        # print searcher_type, [len(scs) for scs in srch_scores]
        _, mean, std = running_max_stats(srch_scores)
        m_maxscores.append(mean)
        std_maxscores.append(std)

    def plot_comparison(k1, k2):
        for searcher_type, mean, std in zip(searchers, m_maxscores, std_maxscores):
            plt.errorbar(np.arange(k1, k2 + 1), mean[k1 - 1:k2], 
                yerr=std[k1 - 1:k2] / np.sqrt(num_repeats), 
                label=searcher_type)
            plt.legend(loc='best')
            plt.xlabel('Number of models evaluated')
            plt.ylabel('Best validation performance')
    
    # plot some different number of ranges
    for k1, k2 in [(1, 64), (1, 16), (4, 16), (16, 64), (6, 64)]:
        plot_comparison(k1, k2)
        plt.savefig(figs_folderpath + 'searcher_comp_%d_to_%d.pdf' % (k1, k2),
             bbox_inches='tight')
        plt.close()

def compute_percentiles(scores, thresholds):
    scores = np.array(scores)
    n = float(len(scores))
    percents = [(scores >= th).sum() / n for th in thresholds]
    return percents

def plot_performance_quantiles():
    folder_path = 'logs/searcher_comparison'
    searchers = ['rand', 'mcts', 'mcts_bi', 'smbo']
    search_space_type = 'deepconv'
    num_repeats = 5

    for searcher_type in searchers:
        # load all the pickles
        rs = []
        for i in xrange(num_repeats):
            file_name = '%s_%s_%d.pkl' % (search_space_type, searcher_type, i)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                r = pickle.load(f)
            rs.append(r)

        percents = np.linspace(0.0, 1.0, num=100)
        srch_scores = [compute_percentiles(r['scores'], percents) 
            for r in rs]
        mean = np.mean(srch_scores, axis=0) 
        std = np.std(srch_scores, axis=0)
        plt.errorbar(percents, mean, 
            yerr=std / np.sqrt(num_repeats), label=searcher_type)

    plt.legend(loc='best')
    plt.xlabel('Validation performance')
    plt.ylabel('Fraction of models better or equal')
    #plt.title('')
    # plt.axis([0, 64, 0.6, 1.0])
    # plt.show()  
    plt.savefig(figs_folderpath + 'quants.pdf', bbox_inches='tight')
    plt.close()

def plot_score_distributions():
    folder_path = 'logs/searcher_comparison'
    searchers = ['rand', 'mcts', 'mcts_bi', 'smbo']
    search_space_type = 'deepconv'
    num_repeats = 5

    m_dists = [] 
    std_dists = []
    for searcher_type in searchers:
        # load all the pickles
        rs = []
        for i in xrange(num_repeats):
            file_name = '%s_%s_%d.pkl' % (search_space_type, searcher_type, i)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                r = pickle.load(f)
            rs.append(r)

        # create the histograms.
        hs = []
        for r in rs:
            h, bin_edges = np.histogram(r['scores'], 50, (0.0, 1.0))
            hs.append(h)
        hs = np.array(hs, dtype='float') / len(r['scores'])
        m_dists.append( np.mean(hs, axis=0) )
        std_dists.append( np.std(hs, axis=0) )
    
    # plot pairwise comparisons with random
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bsrch_type = searchers[0]
    b_mean = m_dists[0]
    b_std = std_dists[0]
    for (srch_type, mean, std) in zip(searchers[1:], m_dists[1:], std_dists[1:]):
        # comparison between random and the other search methods.
        plt.bar(bin_centers, b_mean, yerr=b_std / np.sqrt(num_repeats), 
            width=0.02, alpha=0.8, label=bsrch_type)
        plt.errorbar(bin_centers, b_mean, yerr=b_std / np.sqrt(num_repeats), 
            alpha=0.5, ls='none')
        plt.bar(bin_centers, mean, yerr=std / np.sqrt(num_repeats), 
            width=0.02, alpha=0.8, label=srch_type)
        plt.errorbar(bin_centers, mean, yerr=std / np.sqrt(num_repeats), 
            alpha=0.5, ls='none')
        plt.legend(loc='best')
        plt.xlabel('Validation performance')
        plt.ylabel('Fraction of models')
        plt.savefig(figs_folderpath + 
            'score_dists_%s_vs_%s.pdf' % (bsrch_type, srch_type), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # create the path to output the figures if it does not exists.
    figs_folderpath = 'out/'
    if not os.path.exists(figs_folderpath):
        os.mkdir(figs_folderpath)
    else:
        assert os.path.isdir(figs_folderpath)
            
    plot_searcher_comparison()
    plot_performance_quantiles()
    plot_score_distributions() 
