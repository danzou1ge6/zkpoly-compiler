import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def parse_log_line(line):
    """Parse a log line into structured data."""
    # Fix missing 'T' in Thread if needed
    if line.startswith('hread'):
        line = 'T' + line
        
    # Extract thread ID
    thread_match = re.match(r'[Tt]hread (\d+)', line)
    if not thread_match:
        return None
    thread_id = int(thread_match.group(1))
    
    # Extract instruction type and function name for FuncCall
    func_call_match = re.search(r'instruction FuncCall (\S+)', line)
    if func_call_match:
        instruction_type = 'FuncCall'
        function_name = func_call_match.group(1)
    else:
        instr_match = re.search(r'instruction (\w+)', line)
        if not instr_match:
            return None
        instruction_type = instr_match.group(1)
        function_name = None
    
    # Extract device information
    device_match = re.search(r'device: (\w+) \{ device_id: (\d+) \}', line)
    if device_match:
        device_type = device_match.group(1)
        device_id = int(device_match.group(2))
    else:
        device_type = None
        device_id = None
    
    # Extract stream information
    stream_match = re.search(r'stream: Some\(VariableId\((\d+)\)\)', line)
    if stream_match:
        stream_id = int(stream_match.group(1))
    else:
        stream_id = None
    
    # Extract event information
    event_match = re.search(r'event: EventId\((\d+)\)', line)
    event_id = int(event_match.group(1)) if event_match else None
    
    # Extract variable IDs
    var_ids = []
    var_id_matches = re.finditer(r'VariableId\((\d+)\)', line)
    for match in var_id_matches:
        var_ids.append(int(match.group(1)))
    
    # Extract scalar array info if present
    scalar_match = re.search(r'ScalarArray \{ len: (\d+)', line)
    scalar_len = int(scalar_match.group(1)) if scalar_match else None
    
    # Extract timing information
    start_match = re.search(r'start: (\d+)', line)
    end_match = re.search(r'end: (\d+)', line)
    
    if start_match and end_match:
        start_time = int(start_match.group(1))
        end_time = int(end_match.group(1))
    else:
        return None
    
    return {
        'thread_id': thread_id,
        'instruction_type': instruction_type,
        'function_name': function_name,
        'device_type': device_type,
        'device_id': device_id,
        'stream_id': stream_id,
        'event_id': event_id,
        'var_ids': var_ids,
        'scalar_len': scalar_len,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time
    }

def read_log_file(filename):
    """Read and parse log file into a dataframe."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                data.append(parsed)
    
    return pd.DataFrame(data)

def create_instruction_time_chart(df):
    """Create a bar chart showing time proportions for each instruction type."""
    # Create a column that combines instruction type and function name
    df['operation'] = df.apply(
        lambda x: f"FuncCall-{x['function_name']}" if x['instruction_type'] == 'FuncCall' and x['function_name'] 
        else x['instruction_type'], 
        axis=1
    )
    
    # fuse all the function calls start with "fused_arith" into one
    df['operation'] = df['operation'].replace(to_replace=r'FuncCall-fused_arith.*', value='FuncCall-fused_arith', regex=True)
    # Group by operation and sum durations
    instr_times = df.groupby('operation')['duration'].sum().reset_index()
    # Remove 'Wait' operations
    instr_times = instr_times[instr_times['operation'] != 'Wait']
    total_time = instr_times['duration'].sum()
    instr_times['percentage'] = instr_times['duration'] / total_time * 100
    
    # Sort by duration in descending order
    instr_times = instr_times.sort_values('duration', ascending=False)
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='operation', y='percentage', data=instr_times)
    plt.title('Time Proportion by Instruction Type', fontsize=14)
    plt.xlabel('Instruction Type', fontsize=12)
    plt.ylabel('Percentage of Total Time (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{instr_times["percentage"].iloc[i]:.1f}%',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig('instruction_time_proportions.png', dpi=300)
    plt.close()

def create_gantt_chart(df):
    """Create a Gantt chart showing execution timeline across threads with enhanced color variety."""
    import plotly.graph_objects as go
    from plotly.offline import plot
    import numpy as np
    
    # Create a column that combines instruction type and function name
    df['operation'] = df.apply(
        lambda x: f"FuncCall-{x['function_name']}" if x['instruction_type'] == 'FuncCall' and x['function_name'] 
        else x['instruction_type'], 
        axis=1
    )
    
    # Sort dataframe by thread_id and start_time
    df_sorted = df.sort_values(['thread_id', 'start_time'])
    
    # Get unique threads and operations
    threads = sorted(df_sorted['thread_id'].unique())
    operations = sorted(df_sorted['operation'].unique())
    
    # Create an extended color palette with many more distinct colors
    def generate_distinct_colors(n):
        """Generate a large set of visually distinct colors."""
        # Start with some predefined distinct colors
        distinct_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
            '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
            '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000', '#e6beff', '#ff9a00',
            '#1a8000', '#1ae6e6', '#004d99', '#ff71ce', '#01cdfe', '#05ffa1', '#740070', '#6a0dad',
            '#2f4f4f', '#8b0000', '#006400', '#00008b', '#ff0000', '#ffa500', '#ffff00', '#008000',
            '#0000ff', '#4b0082', '#ee82ee', '#a52a2a', '#deb887', '#5f9ea0', '#7fff00', '#d2691e'
        ]
        
        # If we still need more colors, generate them algorithmically
        if n > len(distinct_colors):
            # Use HSV color space to generate additional distinct colors
            additional_needed = n - len(distinct_colors)
            for i in range(additional_needed):
                # Generate colors evenly spaced around the HSV color wheel
                # Vary saturation and value to increase distinctiveness
                h = i / additional_needed
                s = 0.6 + 0.4 * ((i * 7) % 3) / 2  # Vary saturation
                v = 0.7 + 0.3 * ((i * 11) % 4) / 3  # Vary value
                
                # Convert HSV to RGB
                h_i = int(h * 6)
                f = h * 6 - h_i
                p = v * (1 - s)
                q = v * (1 - f * s)
                t = v * (1 - (1 - f) * s)
                
                if h_i == 0:
                    r, g, b = v, t, p
                elif h_i == 1:
                    r, g, b = q, v, p
                elif h_i == 2:
                    r, g, b = p, v, t
                elif h_i == 3:
                    r, g, b = p, q, v
                elif h_i == 4:
                    r, g, b = t, p, v
                else:
                    r, g, b = v, p, q
                
                # Convert to hex and add to our list
                hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                distinct_colors.append(hex_color)
        
        return distinct_colors[:n]
    
    # Generate colors for each operation
    colors = generate_distinct_colors(len(operations))
    color_dict = {op: colors[i] for i, op in enumerate(operations)}
    
    # Find the minimum start time to normalize timeline
    min_time = df_sorted['start_time'].min()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each operation segment
    for _, row in df_sorted.iterrows():
        thread_id = row['thread_id']
        operation = row['operation']
        start_time = row['start_time'] - min_time  # Normalize time
        end_time = row['end_time'] - min_time      # Normalize time
        duration = end_time - start_time
        
        # Skip very short operations or make them minimal visible width
        if duration < 5:
            end_time = start_time + 5  # Ensure minimal visibility
        
        # Create hover text with detailed information
        hover_text = (f"Thread: {thread_id}<br>"
                     f"Operation: {operation}<br>"
                     f"Start: {row['start_time']}<br>"
                     f"End: {row['end_time']}<br>"
                     f"Duration: {row['duration']}<br>"
                     f"Normalized time: {start_time} to {end_time}")
        
        fig.add_trace(go.Bar(
            x=[end_time - start_time],  # Width of bar represents duration
            y=[f"Thread {thread_id}"],  # Thread label
            orientation='h',
            base=start_time,           # Starting point
            name=operation,            # Name used in legend
            marker_color=color_dict.get(operation, '#CCCCCC'),
            marker=dict(
                line=dict(width=0, color='#000000'),  # Add border to help distinguish
            ),
            text=operation if duration > 50 else "",  # Only show text for longer operations
            textposition='inside',
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False,          # Will add custom legend
            legendgroup=operation      # Group by operation for legend
        ))
    
    # Add a single bar for each operation type in the legend
    added_to_legend = set()
    for _, row in df_sorted.iterrows():
        operation = row['operation']
        if operation not in added_to_legend:
            fig.add_trace(go.Bar(
                x=[0],  # Dummy data
                y=[threads[0] if threads else 0],  # Dummy data
                orientation='h',
                name=operation,
                marker_color=color_dict.get(operation, '#CCCCCC'),
                showlegend=True,
                legendgroup=operation,
                visible=True  # Make legend items visible
            ))
            added_to_legend.add(operation)
    
    # Update layout for better visualization
    fig.update_layout(
        title='Execution Timeline by Thread',
        barmode='overlay',
        bargap=0.1,
        bargroupgap=0.1,
        height=max(600, 100 + len(threads) * 40),  # Dynamic height based on number of threads
        width=1200,
        yaxis=dict(
            title='',
            tickfont=dict(size=14),
            automargin=True,  # Ensure y-axis labels have enough room
        ),
        xaxis=dict(
            title='Time (offset from start)',
            tickformat=',d',  # Format large numbers with commas
            tickfont=dict(size=12),
            tickmode='auto',
            nticks=20,        # More tick marks for better time reference
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            orientation='v',
            itemsizing='constant',
            title=dict(text='Operations')
        ),
        margin=dict(l=250, r=250, t=50, b=50),  # Large left margin for thread labels, right margin for legend
        plot_bgcolor='rgb(248, 248, 248)',
    )
    
    # Add time scale explanation as annotation
    fig.add_annotation(
        x=0,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"Time values are offset from the earliest event in the log (t=0). Raw start time: {min_time}",
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    
    # Save as HTML file
    plot(fig, filename='execution_gantt_chart.html', auto_open=False)
    
    # Create a simpler PNG version for quick reference
    try:
        import plotly.io as pio
        pio.write_image(fig, 'execution_gantt_chart.png', 
                      width=1600, height=max(800, 200 + len(threads) * 40), scale=2)
        print("Gantt charts created: execution_gantt_chart.html (interactive) and execution_gantt_chart.png (static)")
    except Exception as e:
        print(f"Could not create static PNG image: {e}")
        print("Interactive HTML chart was still created successfully.")

def create_function_call_analysis(df):
    """Create analysis specific to function calls."""
    # Filter for function call instructions
    func_calls = df[df['instruction_type'] == 'FuncCall'].copy()
    
    if len(func_calls) == 0:
        print("No function calls found in the log.")
        return
    
    # Group by function name
    func_stats = func_calls.groupby('function_name').agg({
        'duration': ['count', 'mean', 'min', 'max', 'sum']
    }).reset_index()
    
    func_stats.columns = ['function_name', 'count', 'avg_duration', 'min_duration', 'max_duration', 'total_duration']
    func_stats['percentage'] = func_stats['total_duration'] / func_stats['total_duration'].sum() * 100
    
    # Sort by total duration
    func_stats = func_stats.sort_values('total_duration', ascending=False)
    
    # Create bar chart for function call durations
    plt.figure(figsize=(12, 6))
    bars = plt.bar(func_stats['function_name'], func_stats['total_duration'])
    plt.title('Total Execution Time by Function Call', fontsize=14)
    plt.xlabel('Function Name', fontsize=12)
    plt.ylabel('Total Duration', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add duration and percentage labels
    for bar, total, pct in zip(bars, func_stats['total_duration'], func_stats['percentage']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{total}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('function_call_durations.png', dpi=300)
    plt.close()
    
    # Create a table of function call statistics
    plt.figure(figsize=(12, len(func_stats) * 0.5 + 1))
    plt.axis('off')
    
    # Fix: Properly format the data for the table
    # First column is text, others are numeric and should be rounded
    cell_data = []
    for row in func_stats.itertuples(index=False):
        # Convert row to list, keep function_name as is, round numeric values
        formatted_row = [row[0]]  # function_name
        for val in row[1:]:  # numeric columns
            formatted_row.append(round(val, 2) if isinstance(val, (int, float)) else val)
        cell_data.append(formatted_row)
    
    table = plt.table(
        cellText=cell_data,
        colLabels=['Function Name', 'Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Total Duration', 'Percentage (%)'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.savefig('function_call_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_event_record_wait_analysis(df):
    """Analyze Record and Wait instructions to find synchronization patterns."""
    # Filter for Record and Wait instructions
    record_events = df[df['instruction_type'] == 'Record'].copy()
    wait_events = df[df['instruction_type'] == 'Wait'].copy()
    
    if len(record_events) == 0 or len(wait_events) == 0:
        print("Not enough Record/Wait events for analysis")
        return
    
    # Calculate wait times between Record and Wait for same event
    wait_times = []
    for _, wait_row in wait_events.iterrows():
        if wait_row['event_id'] is not None:
            # Find corresponding record for this event
            matching_records = record_events[record_events['event_id'] == wait_row['event_id']]
            if not matching_records.empty:
                latest_record = matching_records.loc[matching_records['end_time'].idxmax()]
                wait_time = wait_row['start_time'] - latest_record['end_time']
                wait_times.append({
                    'event_id': wait_row['event_id'],
                    'record_time': latest_record['end_time'],
                    'wait_time': wait_row['start_time'],
                    'delay': wait_time,
                    'wait_duration': wait_row['duration']
                })
    
    if wait_times:
        wait_df = pd.DataFrame(wait_times)
        
        # Plot wait event delays
        plt.figure(figsize=(10, 6))
        plt.scatter(wait_df['event_id'], wait_df['delay'])
        plt.title('Delay Between Record and Wait for Events')
        plt.xlabel('Event ID')
        plt.ylabel('Delay (time units)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('event_wait_delays.png', dpi=300)
        plt.close()
        
        # Plot wait durations
        plt.figure(figsize=(10, 6))
        plt.scatter(wait_df['event_id'], wait_df['wait_duration'])
        plt.title('Duration of Wait Events')
        plt.xlabel('Event ID')
        plt.ylabel('Wait Duration (time units)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('wait_durations.png', dpi=300)
        plt.close()

def create_stream_activity_chart(df):
    """Create a chart showing activity on different streams."""
    # Filter for instructions with stream information
    stream_ops = df[df['stream_id'].notna()].copy()
    
    if len(stream_ops) == 0:
        print("No stream operations found.")
        return
    
    # Get unique streams
    streams = sorted(stream_ops['stream_id'].unique())
    
    # Create timeline showing each stream's activity
    plt.figure(figsize=(14, 8))
    
    for i, stream_id in enumerate(streams):
        stream_activities = stream_ops[stream_ops['stream_id'] == stream_id]
        for _, activity in stream_activities.iterrows():
            plt.hlines(
                y=i, 
                xmin=activity['start_time'], 
                xmax=activity['end_time'], 
                linewidth=6, 
                color=plt.cm.tab10(i % 10),
                alpha=0.7
            )
            
            # Add operation type as text
            operation = (f"FuncCall-{activity['function_name']}" 
                        if activity['instruction_type'] == 'FuncCall' and activity['function_name'] 
                        else activity['instruction_type'])
            
            mid_point = (activity['start_time'] + activity['end_time']) / 2
            duration = activity['end_time'] - activity['start_time']
            
            # Only add text if there's enough space
            if duration > 15:  
                plt.text(
                    mid_point, 
                    i, 
                    operation, 
                    horizontalalignment='center',
                    verticalalignment='center', 
                    fontsize=8
                )
    
    plt.yticks(range(len(streams)), [f'Stream {int(s)}' for s in streams])
    plt.xlabel('Time')
    plt.title('Stream Activity Timeline')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('stream_activity_timeline.png', dpi=300)
    plt.close()

def visualize_execution_flow(df):
    """Create a visualization of instruction execution flow."""
    # Create a column that combines instruction type and function name
    df['operation'] = df.apply(
        lambda x: f"FuncCall-{x['function_name']}" if x['instruction_type'] == 'FuncCall' and x['function_name'] 
        else x['instruction_type'], 
        axis=1
    )
    
    # Sort by start time
    df_sorted = df.sort_values('start_time').reset_index(drop=True)
    
    # Create transition matrix
    operations = df_sorted['operation'].unique()
    n_ops = len(operations)
    op_to_idx = {op: i for i, op in enumerate(operations)}
    
    transition_matrix = np.zeros((n_ops, n_ops))
    
    for i in range(1, len(df_sorted)):
        prev_op = df_sorted.iloc[i-1]['operation']
        curr_op = df_sorted.iloc[i]['operation']
        prev_idx = op_to_idx[prev_op]
        curr_idx = op_to_idx[curr_op]
        transition_matrix[prev_idx, curr_idx] += 1
    
    # Normalize by row
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=operations,
        yticklabels=operations,
        cbar_kws={'label': 'Transition Probability'}
    )
    plt.title('Instruction Transition Probabilities', fontsize=14)
    plt.xlabel('Next Instruction', fontsize=12)
    plt.ylabel('Current Instruction', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('instruction_transitions.png', dpi=300)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python log_analyze.py <log_file>")
        return
    
    log_file = sys.argv[1]
    
    try:
        # Parse log file
        df = read_log_file(log_file)
        
        # Create visualizations
        create_instruction_time_chart(df)
        create_gantt_chart(df)
        create_function_call_analysis(df)
        create_event_record_wait_analysis(df)
        create_stream_activity_chart(df)
        visualize_execution_flow(df)
        
        print("Visualization complete! Output files:")
        print("1. instruction_time_proportions.png - Bar chart of instruction time proportions")
        print("2. execution_gantt_chart.html - Interactive Gantt chart of execution timeline")
        print("3. function_call_durations.png - Bar chart of function call durations")
        print("4. function_call_statistics.png - Detailed statistics table for function calls")
        print("5. event_wait_delays.png - Analysis of delays between Record and Wait events")
        print("6. wait_durations.png - Analysis of Wait event durations")
        print("7. stream_activity_timeline.png - Timeline showing activity on different streams")
        print("8. instruction_transitions.png - Heatmap of instruction transition probabilities")
        
    except Exception as e:
        print(f"Error processing log file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()