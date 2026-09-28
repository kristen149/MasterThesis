import matplotlib.pyplot as plt

def plot_graphs_pm(stats, file):
    # Pre-init
    percentages_base_u = stats['qualifier1'][1] + stats['qualifier2'][1] + stats['qualifier9'][1]
    percentages_base_p = stats['qualifier1'][3] + stats['qualifier2'][3] + stats['qualifier9'][3]
    percentages_projection_u = stats['qualifier3'][1]
    percentages_projection_p = stats['qualifier3'][3]
    percentages_condition_u = stats['qualifier4'][1]
    percentages_condition_p = stats['qualifier4'][3]

    percentages_base_u = [round(x, 1) for x in percentages_base_u]
    percentages_base_p = [round(x, 1) for x in percentages_base_p]
    percentages_projection_u = [round(x, 1) for x in percentages_projection_u]
    percentages_projection_p = [round(x, 1) for x in percentages_projection_p]
    percentages_condition_u = [round(x, 1) for x in percentages_condition_u]
    percentages_condition_p = [round(x, 1) for x in percentages_condition_p]
    
    fig, (ax, ax1, ax2) = plt.subplots(3, 1,figsize=(14, 8))

    ## GRAPH 1
    # Define data
    categories = ['event\nlevel', 'case \nlevel', 'perspective', 'descr. \nstatistics', 'conformance', 'value', 'generic', 'domain']
    x_pos = [0,1.5, 4,5.5,7, 9.5,11, 12.5]

    bar_width = 0.45

    # Create bar chart
    ax.bar(x_pos, percentages_base_u, width=bar_width, color='lightgray', label='Base utterances')
    ax.bar([i + 0.125 + bar_width for i in x_pos], percentages_base_p, width=bar_width, color='darkgray', label='Paraphrases')
    #ax.bar([i + bar_width*2 for i in x_pos], percentages_base_a, width=bar_width, color='gray', label='Base + Paraphrases')

    #ax1.bar(categories_domain, percentages_domain, width=barWidth, color='gray')
    ax.grid(color='grey', linestyle='--', linewidth=0.15)

    # Set y-axis control ticks
    ax.set_yticks(range(0, 110, 20))

    # Add percentage labels
    for i, counts, in enumerate(percentages_base_u):
        ax.text(x_pos[i], percentages_base_u[i] + 4, counts, ha='center', fontsize=10.5)

    for i, counts, in enumerate(percentages_base_p):
        ax.text(x_pos[i]+ 0.125 + bar_width, percentages_base_p[i] + 4, counts, ha='center', fontsize=10.5)

    #for i, counts, in enumerate(counts_base_a):
    #    ax.text(x_pos[i]+ bar_width*2, percentages_base_a[i] + 1, counts, ha='center', fontsize=6.5)


    # Add title and labels
    # Add labels and title
    ax.set_ylabel('Percentage', fontsize=11)
    #ax.set_title('Process mining qualifiers comparison', fontsize=14)

    # Add legend
    ax.legend(loc="upper center", ncol = 2, bbox_to_anchor=(0.46, 1.8), frameon=False)

    # Set ticks and labels for x-axis
    ax.set_xticks([i + bar_width / 1.55 for i in x_pos])
    ax.set_xticklabels(categories, fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add text
    ax.text(0.4, 120, "Qualifier 1", fontsize=11)
    ax.text(5, 120, "Qualifier 2", fontsize=11)
    ax.text(11, 120, "Qualifier 9", fontsize=11)


    #PROJECTION
    # Define data
    categories_p = ['case', 'event', 'resource', 'activity', 'timestamp', 'cost']
    x_pos_p = [0,1,2,3,4,5]

    bar_width_p = 0.13
    # Create bar chart
    ax1.bar(x_pos_p, percentages_projection_u, width=bar_width_p, color='lightgray', label='Base utterances')
    ax1.bar([i + 0.05 + bar_width_p for i in x_pos_p], percentages_projection_p, width=bar_width_p, color='darkgray', label='Paraphrases')
    #ax.bar([i + bar_width*2 for i in x_pos], percentages_base_a, width=bar_width, color='gray', label='Base + Paraphrases')


    #ax1.bar(categories_domain, percentages_domain, width=barWidth, color='gray')
    ax1.grid(color='grey', linestyle='--', linewidth=0.15)

    # Set y-axis control ticks
    ax1.set_yticks(range(0, 110, 20))

    # Add percentage labels
    for i, counts, in enumerate(percentages_projection_u):
        ax1.text(x_pos_p[i], percentages_projection_u[i] + 4, counts, ha='center', fontsize=10.5)

    for i, counts, in enumerate(percentages_projection_p):
        ax1.text(x_pos_p[i] + 0.05 + bar_width_p, percentages_projection_p[i] + 4, counts, ha='center', fontsize=10.5)

    #for i, counts, in enumerate(counts_base_a):
    #    ax.text(x_pos[i]+ bar_width*2, percentages_base_a[i] + 1, counts, ha='center', fontsize=6.5)


    # Add title and labels
    # Add labels and title
    ax1.set_ylabel('Percentage', fontsize=11)
    #ax.set_title('Process mining projection qualifier', fontsize=14)

    # Add legend
    #ax.legend(loc='upper center', ncol=2)

    # Set ticks and labels for x-axis
    ax1.set_xticks([i + bar_width_p / 1.6 for i in x_pos_p])
    ax1.set_xticklabels(categories_p, fontsize=10)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.text(2, 120, "Qualifier 3", fontsize=11)

    #CONDITION
    categories_w = ['activity', 'none', 'timestamp', 'resource', 'cost', 'case']
    x_pos_w = [0,1,2,3,4,5]

    bar_width_w = 0.13
    # Create bar chart
    ax2.bar(x_pos_w, percentages_condition_u, width=bar_width_w, color='lightgray', label='Base utterances')
    ax2.bar([i + 0.05 + bar_width_w for i in x_pos_w], percentages_condition_p, width=bar_width_w, color='darkgray', label='Paraphrases')
    #ax.bar([i + bar_width*2 for i in x_pos], percentages_base_a, width=bar_width, color='gray', label='Base + Paraphrases')


    #ax1.bar(categories_domain, percentages_domain, width=barWidth, color='gray')
    ax2.grid(color='grey', linestyle='--', linewidth=0.15)

    # Set y-axis control ticks
    ax2.set_yticks(range(0, 110, 20))

    # Add percentage labels
    for i, counts, in enumerate(percentages_condition_u):
        ax2.text(x_pos_w[i], percentages_condition_u[i] + 4, counts, ha='center', fontsize=10.5)

    for i, counts, in enumerate(percentages_condition_p):
        ax2.text(x_pos_w[i]+ 0.05 +bar_width_w, percentages_condition_p[i] + 4, counts, ha='center', fontsize=10.5)

    #for i, counts, in enumerate(counts_base_a):
    #    ax.text(x_pos[i]+ bar_width*2, percentages_base_a[i] + 1, counts, ha='center', fontsize=6.5)


    # Add title and labels
    # Add labels and title
    ax2.set_ylabel('Percentage', fontsize=11)
    #ax.set_title('Process mining condition qualifier', fontsize=14)

    # Add legend


    # Set ticks and labels for x-axis
    ax2.set_xticks([i + bar_width_w / 1.55 for i in x_pos_w])
    ax2.set_xticklabels(categories_w)


    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax2.text(2, 120, "Qualifier 4", fontsize=11)

    # Show plot
    plt.subplots_adjust(hspace=0.95)
    plt.savefig(file, format='png', bbox_inches='tight')


def plot_graphs_nlp(stats, file):
    # Pre-init
    wh_question_u_percentages = stats['qualifier5'][1]
    wh_question_p_percentages = stats['qualifier5'][3]
    
    wh_question_u_percentages = [round(x, 1) for x in wh_question_u_percentages] + [0, 0]
    wh_question_p_percentages = [round(x, 1) for x in wh_question_p_percentages]

    fig, ax = plt.subplots(figsize=(10, 1.5))

    # Define data
    categories = ['how', 'what', 'none', 'which', 'who', 'when']
    x_pos = [0,1,2,3,4,5]

    bar_width = 0.175
    # Create bar chart
    ax.bar(x_pos, wh_question_u_percentages, width=bar_width, color='lightgray', label='Base utterances')
    ax.bar([i + 0.05 + bar_width for i in x_pos], wh_question_p_percentages, width=bar_width, color='darkgray', label='Paraphrases')
    #ax.bar([i + bar_width*2 for i in x_pos], percentages_base_a, width=bar_width, color='gray', label='Base + Paraphrases')


    #ax1.bar(categories_domain, percentages_domain, width=barWidth, color='gray')
    ax.grid(color='grey', linestyle='--', linewidth=0.15)

    # Set y-axis control ticks
    ax.set_yticks(range(0, 110, 20))

    # Add percentage labels
    for i, counts, in enumerate(wh_question_u_percentages):
        ax.text(x_pos[i], wh_question_u_percentages[i] + 4, counts, ha='center', fontsize=8.5)

    for i, counts, in enumerate(wh_question_p_percentages):
        ax.text(x_pos[i]+ 0.05 +bar_width, wh_question_p_percentages[i] + 4, counts, ha='center', fontsize=8.5)

    #for i, counts, in enumerate(counts_base_a):
    #    ax.text(x_pos[i]+ bar_width*2, percentages_base_a[i] + 1, counts, ha='center', fontsize=6.5)


    # Add title and labels
    # Add labels and title
    ax.set_ylabel('Percentage', fontsize=12)
    #ax.set_title('Process mining projection and condition qualifier', fontsize=14)

    # Add legend
    ax.legend(loc="upper center", ncol = 2, bbox_to_anchor=(0.5, 1.8), frameon=False)

    # Set ticks and labels for x-axis
    ax.set_xticks([i + bar_width / 1.5 for i in x_pos])
    ax.set_xticklabels(categories)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.text(2, 110, "Qualifier 5", fontsize=11)

    # Show plot
    plt.savefig(file, format='png', bbox_inches='tight')


def plot_graphs_sql(stats, file):

    # Pre-init
    percentages_group_agg_u = stats['qualifier6'][1] + stats['qualifier7'][1]
    percentages_group_agg_p = stats['qualifier6'][3] + stats['qualifier7'][3]
    percentages_spider_u = stats['qualifier8'][1]
    percentages_spider_p = stats['qualifier8'][3]
    
    percentages_group_agg_u = [round(x, 1) for x in percentages_group_agg_u]
    percentages_group_agg_p = [round(x, 1) for x in percentages_group_agg_p]
    percentages_spider_u = [round(x, 1) for x in percentages_spider_u]
    percentages_spider_p = [round(x, 1) for x in percentages_spider_p]

    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10, 3.5))

    # Define data
    categories_group_agg = ['none', 'aggregation', 'none', 'having']
    x_pos_group_agg = [1,1.5,3,3.5]


    bar_width_g = 0.1
    # Create bar chart
    ax1.bar(x_pos_group_agg, percentages_group_agg_u, width=bar_width_g, color='lightgray', label='Base utterances')
    ax1.bar([i + 0.05 + bar_width_g for i in x_pos_group_agg], percentages_group_agg_p, width=bar_width_g, color='darkgray', label='Paraphrases')
    #ax.bar([i + bar_width*2 for i in x_pos], percentages_base_a, width=bar_width, color='gray', label='Base + Paraphrases')


    #ax1.bar(categories_domain, percentages_domain, width=barWidth, color='gray')
    ax1.grid(color='grey', linestyle='--', linewidth=0.15)

    # Set y-axis control ticks
    ax1.set_yticks(range(0, 110, 20))

    # Add percentage labels
    for i, counts, in enumerate(percentages_group_agg_u):
        ax1.text(x_pos_group_agg[i], percentages_group_agg_u[i] + 4, counts, ha='center', fontsize=8.5)

    for i, counts, in enumerate(percentages_group_agg_p):
        ax1.text(x_pos_group_agg[i]+ 0.05 + bar_width_g, percentages_group_agg_p[i] + 4, counts, ha='center', fontsize=8.5)

    #for i, counts, in enumerate(counts_base_a):
    #    ax.text(x_pos[i]+ bar_width*2, percentages_base_a[i] + 1, counts, ha='center', fontsize=6.5)


    # Add title and labels
    # Add labels and title
    ax1.set_ylabel('Percentage', fontsize=12)
    #ax.set_title('Process mining projection and condition qualifier', fontsize=14)

    # Add legend
    ax1.legend(loc="upper center", ncol = 2, bbox_to_anchor=(0.5, 1.5), frameon=False)

    # Set ticks and labels for x-axis
    ax1.set_xticks([i + bar_width_g / 1.5 for i in x_pos_group_agg])
    ax1.set_xticklabels(categories_group_agg)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.text(1.2, 110, "Qualifier 6", fontsize=11)
    ax1.text(3.2, 110, "Qualifier 7", fontsize=11)


    # Define data
    categories = ['medium', 'no hardness', 'easy', 'hard', 'extra']
    x_pos = [0,1,2,3,4]

    bar_width = 0.15
    # Create bar chart
    ax2.bar(x_pos, percentages_spider_u, width=bar_width, color='lightgray', label='Base utterances')
    ax2.bar([i + 0.05 + bar_width for i in x_pos], percentages_spider_p, width=bar_width, color='darkgray', label='Paraphrases')
    #ax.bar([i + bar_width*2 for i in x_pos], percentages_base_a, width=bar_width, color='gray', label='Base + Paraphrases')


    #ax1.bar(categories_domain, percentages_domain, width=barWidth, color='gray')
    ax2.grid(color='grey', linestyle='--', linewidth=0.15)

    # Set y-axis control ticks
    ax2.set_yticks(range(0, 110, 20))

    # Add percentage labels
    for i, counts, in enumerate(percentages_spider_u):
        ax2.text(x_pos[i], percentages_spider_u[i] + 4, counts, ha='center', fontsize=8.5)

    for i, counts, in enumerate(percentages_spider_p):
        ax2.text(x_pos[i]+ 0.05 + bar_width, percentages_spider_p[i] + 4, counts, ha='center', fontsize=8.5)

    #for i, counts, in enumerate(counts_base_a):
    #    ax.text(x_pos[i]+ bar_width*2, percentages_base_a[i] + 1, counts, ha='center', fontsize=6.5)


    # Add title and labels
    # Add labels and title
    ax2.set_ylabel('Percentage', fontsize=12)
    #ax.set_title('Process mining projection and condition qualifier', fontsize=14)

    # Add legend
    #ax2.legend(loc="upper center", ncols = 2, bbox_to_anchor=(0.5, 1.5), frameon=False)

    # Set ticks and labels for x-axis
    ax2.set_xticks([i + bar_width / 1.5 for i in x_pos])
    ax2.set_xticklabels(categories)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax2.text(1.9, 110, "Qualifier 8", fontsize=11)
    plt.subplots_adjust(hspace=0.55)

    # Show plot
    plt.savefig(file, format='png', bbox_inches='tight')


# prompt: plot group_id on x axis with all labels on axis for each group_id the relation total by EM, bar plot

def plot_metric_groupped(total_group, file):

  bar_width = 0.75

  fig, ax = plt.subplots(figsize=(16, 4))

  ax.bar([i + bar_width for i in total_group['Group_id'].index], total_group['total'], bar_width, color='lightgray', label='Total')
  ax.bar([i + bar_width for i in total_group['Group_id'].index], total_group['EM'], bar_width, color='lightblue', label='Structure Indicator')
  ax.bar([i + bar_width for i in total_group['Group_id'].index], total_group['EX'], bar_width, color='darkred', label='Run Indicator', alpha=0.25)

  ax.set_xlabel('Utterance grouped by base and paraphrase')
  ax.set_ylabel('Total - Structure Indicator - \nRun Indicator')

  ax.set_xticks([])

  ax.axvline(x=37 + bar_width * 0.25, color='black', linestyle='--', linewidth=0.5)
  ax.axvline(x=111 + bar_width * 0.25, color='black', linestyle='--', linewidth=0.5)
  ax.axvline(x=142 + bar_width * 0.25, color='black', linestyle='--', linewidth=0.5)
  ax.axvline(x=156 + bar_width * 0.25, color='black', linestyle='--', linewidth=0.5)

  ax.text(0, 40, "English", fontsize=14)

  ax.text(12 + bar_width, 35, "easy", fontsize=10)
  ax.text(66 + bar_width, 35, "medium", fontsize=10)
  ax.text(123 + bar_width, 35, "hard", fontsize=10)
  ax.text(145 + bar_width, 35, "extra", fontsize=10)
  ax.text(174 + bar_width, 35, "no hardness", fontsize=10)

  ax.grid(color='grey', linestyle='--', linewidth=0.1)

  ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.60, 1.2), frameon=False)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)

  plt.savefig(file, format='png', bbox_inches='tight')