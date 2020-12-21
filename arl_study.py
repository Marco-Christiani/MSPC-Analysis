import mase
import mitten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from datetime import datetime


def generate_dataset_sd(features, sd_shift, N=1000, n_out_control=30):
    """
    Generates a dataset with 1,000 observations
    The last n_out_control observations are out-of-control with sd=sd+sd_shift*sd
    Args:
        features: index of features to be shifted
        sd_shift: sd shift of out-of-control obs as ratio of standard deviation
        N: total number of observations to generate
        n_out_control: number of trailing observations to be shifted
    """
    # Using the full feature set and full covariance matrix
    n_features = 56
    means =  np.repeat(0, n_features)
    full_sim = mase.Simulation(n_observations=N, means=means, 
                            covariance_matrix=np.array(covariance_df))

    # Add Anomalies to feature ----------------------------------------------------
    mean_list = [0] # dont shift mean
    sd_list = [sd_shift]# yes shift sd
    n_obs_list = [n_out_control]

    specs_df = pd.DataFrame(
                data={
                    'mean': mean_list,
                    'sd': sd_list,
                    'n_obs': n_obs_list
                })
    for feature in features:
        full_sim.add_gaussian_observations(specs_df, feature, visualize=False)
    
    # n_out_control = 30
    # n_in_control = n-n_out_control
    # full_df = full_sim.get_data()
    return full_sim.get_data()

def generate_dataset(features, mean_shift, N=1000, n_out_control=30):
    """
    Generates a dataset with 1,000 observations
    The last n_out_control observations are out-of-control with mean=mean+mean_shift*sd
    Args:
        features: index of features to be shifted
        mean_shift: mean shift of out-of-control obs as ratio of standard deviation
        N: total number of observations to generate
        n_out_control: number of trailing observations to be shifted
    """
    # Using the full feature set and full covariance matrix
    n_features = 56
    means =  np.repeat(0, n_features)
    full_sim = mase.Simulation(n_observations=N, means=means, 
                            covariance_matrix=np.array(covariance_df))

    # Add Anomalies to feature ----------------------------------------------------
    mean_list = [mean_shift]
    sd_list = [1]
    n_obs_list = [n_out_control]

    specs_df = pd.DataFrame(
                data={
                    'mean': mean_list,
                    'sd': sd_list,
                    'n_obs': n_obs_list
                })
    for feature in features:
        full_sim.add_gaussian_observations(specs_df, feature, visualize=False)
    
    return full_sim.get_data()


def calc_ucl_from_stats(stats, num_in_control, alpha, step_size=.05):
  """
  HELPER METHOD
  this method is used to calculate a upper control limit for charts given a specified
  false positive rate

  stats: the first component output from any of the mitten control chart methods
  num_in_control: the number of in control rows in the dataset that was fed into the control chart method
                  (should be equal to the parameter passed to the mitten control chart method)
  alpha: the percentage of in control points that will lie below the control line 
          (typically alpha = .05, this is a trade off between false positives and
           a shorter run length time to detect anomalies)
  step_size: parameter that specifies how far the ucl moves each step. As this number
              increases, accuracy and runtime decreases.

  ucl: return the upper control limit
  """

  in_stats = stats[0:num_in_control]
  ucl = max(in_stats)
 
  count = len([i for i in in_stats if i > ucl]) 

  while(count < (alpha* len(in_stats))):
      ucl = ucl - step_size
      count = len([i for i in in_stats if i > ucl]) 
  
  return ucl

def run_length(stats, num_in_control, ucl):
    """
    Calculates the run length using a provided upper control limit

    Args:
        stats: the first component output from any of the mitten control chart methods
        num_in_control: the number of in control rows in the dataset that was fed into the control chart method
                        (should be equal to the parameter passed to the mitten control chart method)
        alpha: the percentage of in control points that will lie below the control line 
                (typically alpha = .05, this is a trade off between false positives and
                a shorter run length time to detect anomalies)
        step_size: parameter that specifies how far the ucl moves each step. As this number
                    increases, accuracy and runtime decreases.

    Returns:
         Run length, the number of out of control observations before an anomaly signal is observed
    """
    out_stats = stats[num_in_control+1:]
    for i in range(len(out_stats)):
        if out_stats[i] > ucl:
            return i+1, ucl
    # No anomaly detected
    return -1

def get_strong_corrs():
    # Correlation groups
    correlation_df = pd.read_csv(data_dir + 'Correlation_Matrix.csv')
    correlation_df = correlation_df.set_index('Row', drop=True)
    # Extract upper triangle since cor(x,y)=cor(y,x)
    correlation_df = correlation_df.where(np.triu(np.ones(correlation_df.shape)).astype(np.bool)) 
    # Set diagonal to 0 since corr(x,x)=1
    for i in range(len(correlation_df)):
        correlation_df.iloc[i,i] = 0 

    corr_pairs = correlation_df.unstack()
    # sorted_pairs = corr_pairs.sort_values(kind="quicksort")
    sorted_pairs = corr_pairs.sort_values(key=lambda r: np.abs(r), ascending=False)
    strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5] # Only show me >50% correlation

    print(strong_pairs)
    features_to_shift = strong_pairs[:3].index.get_level_values(0).values
    features_to_shift = [int(i) for i in features_to_shift]
    print(features_to_shift)

def ARL_Study(mean_shift_list, sims_per_shift=1000, feature_to_shift=6, sd_or_mean_test='MEAN', save=True):
    start = time.time()
    import warnings
    # Stop telling me the covariance matrix is not positive semi-definite please and thank you :)
    warnings.filterwarnings('ignore') 

    sims_run = 0

    hotel_rl_df = pd.DataFrame(columns=mean_shift_list)
    mewma_rl_df = pd.DataFrame(columns=mean_shift_list)
    mcusum_rl_df = pd.DataFrame(columns=mean_shift_list)
    pc_mewma_rl_df = pd.DataFrame(columns=mean_shift_list)
    for mean in mean_shift_list:
        hotel_run_list = []
        mewma_run_list = []
        mcusum_run_list = []
        pc_mewma_run_list = []
        for i in range(sims_per_shift):
            data = generate_dataset([feature_to_shift], mean)
            # data = generate_dataset_sd(features_to_shift, mean) # shifts sd NOT MEAN!!
            n_in_control = 1000-30
            n_out_control = 30
            
            stats, _ = mitten.hotelling_t2(data, n_in_control, plotting=False)  
            run_l, _ = run_length(stats, n_in_control, .01)
            hotel_run_list.append(run_l)

            stats, _ = mitten.apply_mewma(data, n_in_control, plotting=False)
            run_l, _ = run_length(stats, n_in_control, .01)
            mewma_run_list.append(run_l)

            stats, _ = mitten.mcusum(data, n_in_control, 2, plotting=False)
            run_l, _ = run_length(stats, n_in_control, .01)
            mcusum_run_list.append(run_l)

            stats, _ = mitten.pc_mewma(data, n_in_control, 45, plotting=False) # use 45 PC's
            run_l, _ = run_length(stats, n_in_control, .01)
            pc_mewma_run_list.append(run_l)

            sims_run += 1 # cant use walrus bc colab running python 3.6 *sad walrus* :(=
            if sims_run % 10 == 0:
                # print every 10 sims
                sys.stdout.write('\r') # Restart line so output isnt messy 
                sys.stdout.flush()
                print(sims_run, end='') # Overwrite last line
        hotel_rl_df[mean] = hotel_run_list
        mewma_rl_df[mean] = mewma_run_list
        mcusum_rl_df[mean] = mcusum_run_list
        pc_mewma_rl_df[mean] = pc_mewma_run_list
    
    run_time = time.time() - start
    sys.stdout.write('\r') # Restart line
    sys.stdout.flush()
    if save:
        # Save to CSVs with random filenames
        randstring = lambda: '-'.join(np.random.choice(['bill', 'mark', 'kitten', 'mitten', 'mop'],  size=2,
                                                replace=False)) + '-'+datetime.now().strftime('%s') 
        rstr = randstring()
        save_folder = data_dir+f'ARL-{sd_or_mean_test}-{rstr}/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        hotel_rl_df.to_csv(save_folder+f'hotelling-arl.csv', index=False)
        mewma_rl_df.to_csv(save_folder+f'mewma-arl.csv', index=False)
        mcusum_rl_df.to_csv(save_folder+f'mcusum-arl.csv', index=False)
        pc_mewma_rl_df.to_csv(save_folder+f'pc-mewma-arl.csv', index=False)

        # Write documentation file
        # mean_shift_list, sims_per_shift=1000, sd_or_mean_test='MEAN'
        f = open(save_folder+'readme.txt', 'a')
        f.write(f'mean_shift_list: {mean_shift_list}\n')
        f.write(f'sims_per_shift: {sims_per_shift}\n')
        f.write(f'sd_or_mean_test: {sd_or_mean_test}\n')
        f.write('-'*50)
        f.write(f'\nRan a total of {sims_run} simulations spanning {len(mean_shift_list)} mean shifts\n')
        f.write(f'Seconds to run: {run_time}\n')
        f.write(f'Secs/Sim: {run_time/sims_run}\n')
        f.close()
        print(f'CSVs saved: {save_folder}\n')
    else:
        print('CSVs not saved.\n')
    
    print(f'Ran a total of {sims_run} simulations spanning {len(mean_shift_list)} mean shifts')
    print('Seconds to run:', run_time)
    print('Secs/Sim:', run_time/sims_run)
    return hotel_rl_df, mewma_rl_df, mcusum_rl_df, pc_mewma_rl_df


def Diagnostic_Study(mean_shift_list, features_to_shift, sims_per_shift=1000, save=True):
    def rank_of_correct(correct_feature, ranks):
        return ranks.values[ranks.index == correct_feature][0]+1
    start = time.time()
    import warnings
    # Stop telling me the covariance matrix is not positive semi-definite please and thank you :)
    warnings.filterwarnings('ignore') 

    sims_run = 0
    nrows = range(len(features_to_shift)*sims_per_shift)
    hotel_rank_df = pd.DataFrame(index=nrows, columns=mean_shift_list)
    mewma_rank_df = pd.DataFrame(index=nrows, columns=mean_shift_list)
    mcusum_rank_df = pd.DataFrame(index=nrows, columns=mean_shift_list)
    pc_mewma_rank_df = pd.DataFrame(index=nrows, columns=mean_shift_list)
    for mean in mean_shift_list:
        row_idx = 0
        for feature_to_shift in features_to_shift:
            for i in range(sims_per_shift):
                data = generate_dataset(features_to_shift, mean)
                # data = generate_dataset_sd(features_to_shift, mean) # shifts sd NOT MEAN!!
                n_in_control = 1000-30
                n_out_control = 30
                
                stats, ucl = mitten.hotelling_t2(data, n_in_control, plotting=False)  
                run_l = run_length(stats, n_in_control, ucl)
                t_rank_srs = mitten.interpret_multivariate_signal(data, stats, ucl, batch_size=3)
                t_rank = rank_of_correct(feature_to_shift, t_rank_srs) # find where correct culprit was ranked (on average)
                hotel_rank_df[mean][row_idx] = t_rank

                stats, ucl = mitten.apply_mewma(data, n_in_control, plotting=False)
                run_l = run_length(stats, n_in_control, ucl)
                t_rank_srs = mitten.interpret_multivariate_signal(data, stats, ucl, batch_size=3)
                t_rank = rank_of_correct(feature_to_shift, t_rank_srs) # find where correct culprit was ranked (on average)
                mewma_rank_df[mean][row_idx] = t_rank

                stats, ucl = mitten.mcusum(data, n_in_control, 2, plotting=False)
                run_l = run_length(stats, n_in_control, ucl)
                t_rank_srs = mitten.interpret_multivariate_signal(data, stats, ucl, batch_size=3)
                t_rank = rank_of_correct(feature_to_shift, t_rank_srs) # find where correct culprit was ranked (on average)
                mcusum_rank_df[mean][row_idx] = t_rank

                stats, ucl = mitten.pc_mewma(data, n_in_control, 45, plotting=False) # use 45 PC's
                run_l = run_length(stats, n_in_control, ucl)
                t_rank_srs = mitten.interpret_multivariate_signal(data, stats, ucl, batch_size=3)
                t_rank = rank_of_correct(feature_to_shift, t_rank_srs) # find where correct culprit was ranked (on average)
                pc_mewma_rank_df[mean][row_idx] = t_rank

                sims_run += 1 # cant use walrus bc colab running python 3.6 *sad walrus* :(=
                if sims_run % 10 == 0: # print every 10 sims
                    sys.stdout.write('\r') # Overwrite last line/restart line so output isnt messy 
                    sys.stdout.flush()
                    print(sims_run, end='')
                row_idx += 1
    
    run_time = time.time() - start
    sys.stdout.write('\r') # Overwrite last line/restart line
    sys.stdout.flush()
    if save:
        # Save to CSVs with random filenames
        randstring = lambda: '-'.join(np.random.choice(['bill', 'mark', 'kitten', 'mitten', 'mop'],  size=2,
                                                replace=False)) + '-'+datetime.now().strftime('%s') 
        rstr = randstring()
        save_folder = data_dir+f'Diagnostic-{rstr}/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        hotel_rank_df.to_csv(save_folder+f'hotelling-diagnostic.csv', index=False)
        mewma_rank_df.to_csv(save_folder+f'mewma-diagnostic.csv', index=False)
        mcusum_rank_df.to_csv(save_folder+f'mcusum-diagnostic.csv', index=False)
        pc_mewma_rank_df.to_csv(save_folder+f'pc-mewma-diagnostic.csv', index=False)

        plt.figure(figsize=(8,6))
        plt.plot(hotel_rank_df.mean(), '.-g')
        plt.plot(mewma_rank_df.mean(), '.-b')
        plt.plot(mcusum_rank_df.mean(), '.-r')
        plt.plot(pc_mewma_rank_df.mean(), '.-')

        plt.xticks(shift_list)
        plt.legend(['Hotelling', 'MEWMA', 'MCUSUM', 'PC-MEWMA'])
        plt.xlabel('Mean Shift Size (as ratio of $\sigma$)')
        plt.ylabel('Average Ranking of Culprit Feature')
        plt.title('Ranked T-Statistic Diagnostic Test')
        plt.savefig(save_folder+'t-stat-diagnostic-plot.png', dpi=300)

        # Write documentation file
        # mean_shift_list, sims_per_shift=1000, sd_or_mean_test='MEAN'
        f = open(save_folder+'readme.txt', 'a')
        f.write(f'mean_shift_list: {mean_shift_list}\n')
        f.write(f'sims_per_shift: {sims_per_shift}\n')
        f.write('-'*50)
        f.write(f'\nRan a total of {sims_run} simulations spanning {len(mean_shift_list)} mean shifts\n')
        f.write(f'Seconds to run: {run_time}\n')
        f.write(f'Secs/Sim: {run_time/sims_run}\n')
        f.close()
        print(f'CSVs saved: {save_folder}\n')
    else:
        print('CSVs not saved.\n')

    print(f'Ran a total of {sims_run} simulations spanning {len(mean_shift_list)} mean shifts')
    print('Seconds to run:', run_time)
    print('Secs/Sim:', run_time/sims_run)
    hotel_rank_df = hotel_rank_df.reset_index(drop=True)
    mewma_rank_df = mewma_rank_df.reset_index(drop=True)
    mcusum_rank_df = mcusum_rank_df.reset_index(drop=True)
    pc_mewma_rank_df = pc_mewma_rank_df.reset_index(drop=True)
    return hotel_rank_df, mewma_rank_df, mcusum_rank_df, pc_mewma_rank_df


if __name__ == '__main__':
    data_dir = '<my_dir>'
    covariance_df = pd.read_csv(data_dir + 'Covariance_Matrix.csv')
    covariance_df = covariance_df.set_index('Row', drop=True)

    shift_list = np.linspace(0.5, 2, 4)
    hotel_rl_df, mewma_rl_df, mcusum_rl_df, pc_mewma_rl_df = ARL_Study(shift_list, sims_per_shift=100, save=True)

    plt.style.use('ggplot')
    plt.figure(figsize=(8,6))
    plt.plot(hotel_rl_df.mean(), '.-g')
    plt.plot(mewma_rl_df.mean(), '.-b')
    plt.plot(mcusum_rl_df.mean(), '.-r')
    plt.plot(pc_mewma_rl_df.mean(), '.-', color='gray')
    plt.xticks(np.linspace(0.1,2,20))
    plt.legend(['Hotelling', 'MEWMA', 'MCUSUM', 'PC-MEWMA'])

    shift_list = np.linspace(0.5, 2, 4)
    hotel_rank_df, mewma_rank_df, mcusum_rank_df, pc_mewma_rank_df = Diagnostic_Study(shift_list, [6], sims_per_shift=100, save=True)

    plt.figure(figsize=(8,6))
    plt.plot(hotel_rank_df.mean(), '.-g')
    plt.plot(mewma_rank_df.mean(), '.-b')
    plt.plot(mcusum_rank_df.mean(), '.-r')
    plt.plot(pc_mewma_rank_df.mean(), '.-')

    # plt.yticks(list(plt.yticks()[0]) + [3])
    plt.xticks(shift_list)
    plt.legend(['Hotelling', 'MEWMA', 'MCUSUM', 'PC-MEWMA'])
    plt.xlabel('Mean Shift Size (as ratio of $\sigma$)')
    plt.ylabel('Average Ranking of Culprit Feature')
    plt.title('Ranked T-Statistic Diagnostic Test')
    plt.savefig(data_dir+'t-stat-diagnostic-plot.png', dpi=300)
