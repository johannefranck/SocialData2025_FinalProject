import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
pd.set_option('future.no_silent_downcasting', True)


# ----------------------------- bar percentage distribution plot  ----------------------------- #
def plot_bar_dist_perc(values, percentage, html, color, y_axis_title, title):
    values = [str(v) for v in values]
    plot_df = pd.DataFrame({'Value': values, 'Percentage': percentage})
    plot_df['Value'] = pd.Categorical(plot_df['Value'], categories=values, ordered=True)

    fig = px.bar(plot_df, x='Value', y='Percentage', text='Percentage')

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker_color=color,
        marker_line_color='black',
        marker_line_width=1.5,
        hovertemplate=(
            "Year: %{x}<br>" +
            ("Percentage: %{y:.2f}%"  +
            "<extra></extra>"))
        )

    fig.update_layout(
        title=title,
        title_x=0.5,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title=y_axis_title,
        xaxis_title="Election Year",
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.write_html(html)
    fig.show()

# ----------------------------- Stacked-bar plot  ----------------------------- #
def plot_stacked_bar_dist(years, counts, legend, values, html, color_map, show_percentage, y_axis_title, title):
    df = pd.DataFrame({'Year': years, f'{legend}': counts, 'Value': values})
    df['Year'] = df['Year'].astype(str)
    df[f'{legend}'] = df[f'{legend}'].astype(str)

    if show_percentage:
        df['Percentage'] = df.groupby('Year')['Value'].transform(lambda x: x / x.sum() * 100)
        y_col = 'Percentage'
        text_format = '%{text:.1f}%'
    else:
        y_col = 'Value'
        text_format = '%{text:.0f}'

    if isinstance(color_map, list):
        fig = px.bar(
            df,
            x='Year',
            y=y_col,
            color=f'{legend}',
            text=y_col,
            barmode='stack',
            color_discrete_sequence=color_map  
        )
    else:
        fig = px.bar(
            df,
            x='Year',
            y=y_col,
            color=f'{legend}',
            text=y_col,
            barmode='stack',
            color_discrete_map=color_map  
        )
        
    fig.update_traces(
        texttemplate=text_format,
        textposition='inside',
        marker_line_color='lightgrey',
        marker_line_width=1.5, 
        hovertemplate=(
            "Year: %{x}<br>" +
            ("Percentage: %{y:.2f}%" if show_percentage else "Count: %{y:.0f}") +
            "<extra></extra>")
        )

    fig.update_layout(
        title=title,
        title_x=0.5,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title=y_axis_title,
        xaxis_title="Election Year"
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.write_html(html)
    fig.show()

# ----------------------------- Pyramid population plot  ----------------------------- #
def plot_population_pyramids(
    df, 
    age_col='Age', 
    gender_col='Gender', 
    year_col='Year', 
    count_col='Count', 
    genders=('Men', 'Women'), 
    colors=('blue', 'red'), 
    max_years=6, 
    suptitle="Population Pyramids by Year"
):
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Clean age column
    df = df.copy()
    df[age_col] = df[age_col].str.replace('år', '').str.strip()
    age_order = df[age_col].unique()

    # Filter gender and age
    df_filtered = df[
        df[gender_col].isin(genders) & 
        df[age_col].isin(age_order)
    ]

    # Get colors
    color_m, color_k = colors

    # Setup subplots
    years = sorted(df_filtered[year_col].unique())[:max_years]
    n_cols = 2
    n_rows = (len(years) + 1) // 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9 * n_cols, 5.5 * n_rows))
    axs = axs.flatten()

    # Global max for x-axis scaling
    max_count = 0
    for year in years:
        df_year = df_filtered[df_filtered[year_col] == year]
        pivot = df_year.groupby([age_col, gender_col])[count_col].sum().unstack().fillna(0)
        max_count = max(max_count, pivot.values.max())

    # Plot pyramids
    for i, year in enumerate(years):
        ax = axs[i]
        df_year = df_filtered[df_filtered[year_col] == year]
        pop = df_year.groupby([age_col, gender_col])[count_col].sum().unstack().fillna(0)
        pop = pop.loc[age_order]

        ax.barh(pop.index, -pop.get(genders[0], 0), color=color_m, label=genders[0])
        ax.barh(pop.index, pop.get(genders[1], 0), color=color_k, label=genders[1])

        ax.set_title(f"{year}", fontsize=14)
        ax.set_ylabel('Age Group')
        ax.set_xlabel('Population')
        ax.set_xlim(-max_count, max_count)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"{int(abs(label))}" for label in ax.get_xticks()])

        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.suptitle(suptitle, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# ----------------------------- Line plot by year  ----------------------------- #
def line_plot_by_year(df, x, y, cat, title, x_title, y_title, color_map, html): 
    # Sort by your desired district order
    district_order = list(color_map.keys())
    df[cat] = pd.Categorical(df[cat], categories=district_order, ordered=True)
    df = df.sort_values([cat, x])

    # Initialize figure
    fig = go.Figure()

    # Add each district as a separate trace
    for kreds_name in district_order:
        kreds_df = df[df['KredsNr'] == kreds_name]

        fig.add_trace(
            go.Scatter(
                x=kreds_df[x],
                y=kreds_df[y],
                mode='lines+markers',
                name=kreds_name,
                marker=dict(color=color_map[kreds_name]),
                line=dict(color=color_map[kreds_name]),
                hovertemplate=(
                    f"<b>{kreds_name}</b><br>" +
                    "Population Count = %{y}<extra></extra>"
                )
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, 
            font=dict(
                family="Arial",  
                size=20,         
                color="black")),
        font=dict(
                family="Arial",
                size=12, 
                color="black"),
        legend=dict(
                title_font_family="Arial",
                font=dict(size=12),
                orientation="v",  
                traceorder="normal"),
        xaxis_title=x_title, 
        yaxis_title=y_title,
        template='plotly',
        hovermode="x unified",
    )

    fig.write_html(html)
    fig.show()
    
# ----------------------------- Grouped percentage bar plot  ----------------------------- #   
def plot_grouped_percentage_bar(
    df,
    x_col,
    y_col,
    group_col,
    color_sequence,
    html,
    category_order=None,
    title="",
    y_axis_title="Share of Households (%)",
    x_axis_title="Year",
):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=group_col,
        color_discrete_sequence=color_sequence,
        barmode='group',
        category_orders={group_col: category_order} if category_order else None,
        text=y_col
    )

    # Text formatting and hover
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside',
        hovertemplate="<b>%{fullData.name}</b><br>" +
                      f"{x_axis_title} = %{{x}}<br>" +
                      f"{y_axis_title}: %{{y:.2f}}%<extra></extra>"
    )

    # Layout styling
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(family="Arial", size=20, color="black")
        ),
        font=dict(family="Arial", size=12, color="black"),
        barmode='group',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title=y_axis_title,
        xaxis_title=x_axis_title,
        legend=dict(
            title_text=None,
            title_font_family="Arial",
            font=dict(size=12),
            orientation="v",
            traceorder="normal"
        ),
        xaxis=dict(
            categoryorder='array',
            categoryarray=sorted(df[x_col].unique()),
            showline=True,
            linewidth=1,
            linecolor='lightgrey',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='lightgrey',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
    )
    fig.write_html(html)
    fig.show()

# ----------------------------- Bar plot per constituency  ----------------------------- #
def plot_bar_dist_per_const_multi(years, kreds_ids, values, html, const_colors, constituency_id_to_name, show_percentage=False, title="", y_axis_title=""):
    # Create the DataFrame
    df = pd.DataFrame({
        'Year': [str(y) for y in years],
        'KredsNr': kreds_ids,
        'Value': values})

    # Map KredsNr to name
    df['KredsName'] = df['KredsNr'].map(constituency_id_to_name)

    df['Percentage'] = df['Value'] * 100
    y_col = 'Percentage'
    text_format = '%{text:.1f}%'
    y_title = y_axis_title

    # Create traces for each Kreds
    kreds_list = []
    for kreds_name in const_colors.keys():
        kreds_nr = df[df['KredsName'] == kreds_name]['KredsNr'].iloc[0]
        kreds_list.append((kreds_nr, kreds_name))

    fig = go.Figure()

    for kreds_nr, kreds_name in kreds_list:
        kreds_df = df[df['KredsName'] == kreds_name]

        fig.add_trace(
            go.Bar(
                x=kreds_df['Year'],
                y=kreds_df[y_col],
                name=kreds_name,
                text=kreds_df[y_col],
                marker_color=const_colors[kreds_name],
                texttemplate=text_format,
                textposition='inside',
                visible=True, 
                hovertemplate=(
                    f"<b>{kreds_name}</b><br>" +
                    f"{'Percentage' if show_percentage else 'Count'}: %{{y:.2f}}<extra></extra>")
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, 
            font=dict(
                family="Arial",  
                size=20,         
                color="black")),
        barmode='group',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title=y_title,
        xaxis_title="Election Year",
        font=dict(
            family="Arial",
            size=12, 
            color="black"),
        legend=dict(
            title_font_family="Arial",
            font=dict(size=12),
            bordercolor="Black",
            borderwidth=1,
            orientation="v",  
            traceorder="normal")
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')
    
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor='lightgrey',
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        range=[0, 100] if show_percentage else None)
    

    fig.write_html(html)
    fig.show()

# ----------------------------- Stacked bar plot per constituency  ----------------------------- #
def plot_stacked_bar_dist_per_const(years, kreds_ids, counts, legend, values, html, color_map, const_order, constituency_id_to_name, show_percentage=False, title="", y_title=""):
    df = pd.DataFrame({
        'Year': years,
        'KredsNr': kreds_ids,
        f'{legend}': counts,
        'Value': values})

    # Map KredsNr to KredsName
    df['KredsName'] = df['KredsNr'].map(constituency_id_to_name)

    # Convert years to string for categorical x-axis
    df['Year'] = df['Year'].astype(str)
    df[f'{legend}'] = df[f'{legend}'].astype(str)

    # Compute percentage if needed
    if show_percentage:
        df['Percentage'] = df['Value'] * 100
        y_col = 'Percentage'
        text_format = '%{text:.1f}%'
        y_title = y_title
    else:
        y_col = 'Value'
        text_format = '%{text:.0f}'
        y_title = y_title

    # Get sorted list of (KredsNr, KredsName)
    kreds_list = []
    for kreds_name in const_order:
            kreds_nr = df[df['KredsName'] == kreds_name]['KredsNr'].iloc[0]
            kreds_list.append((kreds_nr, kreds_name))

    fig = go.Figure()
    # Category list : parties 
    all_categories = sorted(df[legend].unique())

    # Create traces for each (KredsName, Category) combination
    for kreds_nr, kreds_name in kreds_list:
        kreds_df = df[df['KredsName'] == kreds_name]

        for cat in all_categories:
            cat_df = kreds_df[kreds_df[legend] == cat]

            fig.add_trace(
                go.Bar(
                    x=cat_df['Year'],
                    y=cat_df[y_col],
                    name=cat, 
                    legendgroup=cat,  
                    marker_color=color_map.get(cat, '#333333'),
                    text=cat_df[y_col],
                    texttemplate=text_format,
                    textposition='inside',
                    visible=(kreds_name == 'Østerbro'),  
                    hovertemplate=(
                    f"<b>{cat}</b><br>" +
                    f"{'Percentage' if show_percentage else 'Count'}: %{{y:.2f}}<extra></extra>")
                )
            )

    # Checklist (checkbox-style menu)
    num_categories = len(all_categories)
    kreds_names_only = [k for _, k in kreds_list]

    buttons = []
    for i, kreds_name in enumerate(kreds_names_only):
        visible = [False] * len(fig.data)
        start_idx = i * num_categories

        for j in range(num_categories):
            visible[start_idx + j] = True

        buttons.append(dict(
            label=kreds_name,
            method="update",
            args=[
                {"visible": visible},
                {"title": f"{title}<br>({kreds_name})"}  
            ]
        ))
    default = kreds_names_only[0]
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=1.05,  
                y=1.25, 
                active=0,  
                showactive=True,
                bgcolor="white",
                bordercolor="black",
                borderwidth=0.5,
                font=dict(size=12),
                direction="down"
            )
        ],
        title=dict(
            text=f"{title}<br>({default})",  
            x=0.5, 
            font=dict(
                family="Arial",  
                size=20,         
                color="black")),
        barmode='stack',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title=y_title,
        xaxis_title="Election Year",
        font=dict(
            family="Arial",
            size=12, 
            color="black"),
        showlegend=True,
        legend_title_text=legend,
        legend=dict(
            title_font_family="Arial",
            font=dict(size=12))
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.write_html(html)
    fig.show()

# ------------------------------ Line plot per party over time  ----------------------------- #
def plot_party_vote_share_over_time(df, party_colors, output_path):
    years = sorted(df['Year'].unique())

    fig = px.line(
        df,
        x='Year',
        y='VoteShare',
        color='Partyname',
        markers=True,
        title='Vote Share per Party Over Time (2005-2022)',
        labels={'VoteShare': 'Vote Share (%)', 'Year': 'Election Year'},
        color_discrete_map=party_colors
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=years,
        ticktext=[str(y) for y in years]
    )

    fig.update_layout(
        title=dict(
            text='Vote Share by Party Over Time (2005-2022)',
            x=0.5,
            font=dict(family="Arial", size=20, color="black")
        ),
        font=dict(family="Arial", size=12, color="black"),
        yaxis_tickformat='.0%',
        hovermode='x unified',
        legend_title_text='Party',
        legend=dict(
            title_font_family="Arial",
            font=dict(size=12),
            orientation="v",
            traceorder="normal"
        )
    )

    fig.update_traces(
        hovertemplate='%{fullData.name}: %{y:.1%}<extra></extra>'
    )

    fig.write_html(output_path)
    fig.show()

# ----------------------------- Dropdown plot per constituency  ----------------------------- #    
def plot_party_vote_share_dropdown(
    df,
    party_colors,
    constituency_name_to_id,
    const_order,
    output_path="html_plots/party_vote_share_const_dropdown.html"
):
    df = df.copy()
    df['Year'] = df['Year'].astype(str)

    const_names = const_order
    const_idxs = [constituency_name_to_id[name] for name in const_names]

    # Sort parties by first appearance year
    parties_sorted = (
        df[df['Votes (%)'] > 0]
        .groupby('Partyname')['Year']
        .min()
        .astype(int)
        .sort_values()
    )
    parties = parties_sorted.index.tolist()

    fig = go.Figure()
    n_parties = len(parties)

    # Add bar traces for each constituency and party
    for c_idx, (const_id, cname) in enumerate(zip(const_idxs, const_names)):
        sub = df[df['KredsNr'] == const_id]
        for party in parties:
            tmp = (
                sub[sub['Partyname'] == party]
                .pivot(index='Year', columns='Partyname', values='Votes (%)')
                .fillna(0)
                .reset_index()
            )

            fig.add_trace(
                go.Bar(
                    x=tmp['Year'],
                    y=tmp[party],
                    name=party,
                    marker_color=party_colors.get(party, "#333333"),
                    visible=(c_idx == 0),
                    offsetgroup=party,
                    legendgroup=party,
                    text=tmp[party].round(1),
                    textposition="inside",
                    textfont=dict(color="white", size=12),
                    hovertemplate=(
                        f"<b>{cname}</b><br>"
                        "Year: %{x}<br>"
                        f"Party: {party}<br>"
                        "Vote share: %{y:.1f}%<extra></extra>"
                    )
                )
            )

    # Create dropdown buttons for constituencies
    buttons = []
    for c_idx, cname in enumerate(const_names):
        vis = (
            [False] * n_parties * c_idx +
            [True] * n_parties +
            [False] * n_parties * (len(const_names) - c_idx - 1)
        )
        buttons.append(
            dict(
                label=cname,
                method="update",
                args=[
                    {"visible": vis},
                    {"title.text": f"Party Vote Shares Across Elections in {cname}"}
                ]
            )
        )

    # Layout settings
    fig.update_layout(
        title=dict(
            text=f"Party Vote Shares Across Elections in {const_names[0]}",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Election Year",
        yaxis_title="Vote Share (%)",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.04,
        font=dict(family="Arial"),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=1.02, xanchor="left",
            y=1.2, yanchor="top",
            showactive=True
        )],
        legend=dict(title="Party", font_size=12),
        margin=dict(l=40, r=20, t=50, b=40)
    )

    fig.write_html(output_path)
    fig.show()


# ----------------------------- Trend line plot per constituency  ----------------------------- #
##### NB! I skal ændre denne funktion til at bruge keys af const_colors til at få order også skal den have den som input, se de andre funktioner 


const_colors = {
  'Indre By':     '#241B5F',
  'Østerbro':     '#46237A',
  'Vesterbro':    '#4A4DE9',
  'Nørrebro':     '#ABD2FA',
  'Bispebjerg':   '#FFB400',
  'Brønshøj':      '#FFC800',
  'Valby':        '#FFE5B4',
  'Falkoner':     '#FF5A00',
  'Slots':         '#FF8427',
  'Sundbyvester': '#D64161',
  'Sundbyøster':  '#FF6273',
  'Tårnby':       '#FF95A1'}

const_order = ["Indre By", "Østerbro", "Vesterbro", "Nørrebro", "Bispebjerg", "Brønshøj", "Valby", "Falkoner", "Slots", "Sundbyvester", "Sundbyøster", "Tårnby"]

def plot_stacked_line_from_grouped_df(
    df_grouped,
    colors,
    constituency_id_to_name,
    show_percentage=False,
    title="",
    y_axis_title="",
    group_name="",
    group_order=[]
):

    column_names = df_grouped.columns.to_list()

    # Prepare DataFrame
    df = df_grouped.copy()
    df['KredsName'] = df['KredsNr'].map(constituency_id_to_name)
    df['Year'] = df['Year'].astype(str)
    df['Year'] = pd.Categorical(df['Year'], categories=sorted(df['Year'].unique()), ordered=True)

    # Calculate percentage per year and kreds if needed
    if show_percentage:
        df['DisplayValue'] = df.groupby(['KredsName', 'Year'])['Count'].transform(lambda x: x / x.sum() * 100)
    else:
        df['DisplayValue'] = df['Count']

    # Define age group order explicitly
    constituencies = df['KredsName'].dropna().unique()

    fig = go.Figure()

    # Add traces
    for i, constituency in enumerate(constituencies):
        for group in group_order:
            sub = df[(df['KredsName'] == constituency) & (df[column_names[2]] == group)]
            sub = sub.sort_values('Year')

            fig.add_trace(go.Scatter(
                x=sub['Year'],
                y=sub['DisplayValue'],
                mode='lines',
                stackgroup='one',
                name=group,
                marker=dict(color=const_colors.get(group)),
                line_shape="linear",
                visible=(i == 0),
                hovertemplate=(
                    f"<b>{group}</b><br>Year: %{{x}}<br>"
                    f"{'%' if show_percentage else 'Count'}: %{{y:.1f}}<extra></extra>")
            ))

    # Dropdown buttons
    buttons = []
    for i, constituency in enumerate(constituencies):
        visibility = []
        for j in range(len(constituencies)):
            visibility.extend([(j == i)] * len(group_order))
        buttons.append(dict(
            label=constituency,
            method='update',
            args=[{"visible": visibility},
                  {"title": f"{title} - {constituency}"}]
        ))

    # Layout
    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.1, yanchor="top"
        )],
        title=dict(text=f"{title} - {constituencies[0]}", x=0.5),
        xaxis_title="Year of Data Collection",
        yaxis_title=y_axis_title if not show_percentage else "Percentage",
        xaxis=dict(type='category'),
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        legend_title_text=group_name,
        legend=dict(orientation="v", bordercolor="Black", borderwidth=1)
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey',
                     showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.show()


# ----------------------------- Support layerede plot per constituency  ----------------------------- #
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Title, Legend
from bokeh.io import output_file, save
from math import pi

def plot_support_by_district(df_grouped, const_order, expanded_theme_colors, html=None):
    # Compute the mean percentage over all years for each District + SupportType
    df_support_mean = (
        df_grouped
        .groupby(["District", "SupportType"], observed=True)["Percentage"]
        .mean()
        .reset_index()
    )

    # Ensure only relevant districts and correct order
    df_support_mean = df_support_mean[df_support_mean["District"].isin(const_order)]
    df_support_mean["District"] = pd.Categorical(df_support_mean["District"], categories=const_order, ordered=True)

    # Pivot to wide format for plotting
    df_ratio = df_support_mean.pivot(index="District", columns="SupportType", values="Percentage").reset_index()
    df_ratio = df_ratio.sort_values("District")

    # Support types and matching color indices
    support_types_layered_order = [
        "State Pension",
        "Unemployment Benefits",
        "Social Assistance (Cash Benefits)",
        "Health / Disability Support",
        "Parental Leave Benefits",
        "Disability Pension",
        "Early Retirement Pension",
        "Activation Programs"
    ]
    support_color_indices = [13, 0, 4, 6, 1, 10, 3, 8]
    support_colors_ordered = [expanded_theme_colors[i] for i in support_color_indices]

    source = ColumnDataSource(df_ratio)

    # Create Bokeh figure
    p = figure(
        x_range=const_order,
        title="Support Type Share by District",
        width=900, height=600,
        toolbar_location="right",
        tools="pan,box_zoom,reset,save"
    )

    subtitle = Title(
        text="Mean over years [2009,2014,2019]",
        text_font="Arial",
        text_font_size="12pt",
        text_color="#444444",
        text_font_style="normal"
    )
    p.add_layout(subtitle, 'above')

    # Fonts
    p.title.text_font = "Arial"
    p.title.text_font_size = "20pt"
    p.title.text_font_style = "normal"

    p.xaxis.axis_label_text_font = "Arial"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.axis_label_text_font_style = "normal"
    p.xaxis.major_label_text_font = "Arial"
    p.xaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_style = "normal"

    p.yaxis.axis_label_text_font = "Arial"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_style = "normal"
    p.yaxis.major_label_text_font = "Arial"
    p.yaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_style = "normal"

    # Add bars (layered)
    legend_items = []
    for color, support in zip(support_colors_ordered, support_types_layered_order):
        r = p.vbar(
            x="District",
            top=support,
            source=source,
            width=0.7,
            color=color,
            alpha=0.9,
            muted_alpha=0.1
        )
        legend_items.append((support, [r]))

    legend = Legend(
        items=legend_items,
        location="center",
        orientation="horizontal",
        label_text_font="Arial",
        label_text_font_size="12pt",
        label_text_font_style="normal",
        click_policy="hide",
        spacing=10,
        ncols=4
    )
    p.add_layout(legend, 'below')

    # Axes
    p.xaxis.major_label_orientation = -pi / 4
    p.yaxis.axis_label = "Share of Population (%)"
    p.xaxis.axis_label = "District"

    # Save if path is given
    if html:
        output_file(html)
        save(p)

    # Always show in notebook
    show(p)
