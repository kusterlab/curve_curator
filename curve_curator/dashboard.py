# dashboard.py
# Data analysis platform based on interactive Bokeh plots with HTML/JS output.
#
# Florian P. Bayer - 2024
#

import numpy as np
import pandas as pd

from . import toolbox as tool
from . import user_interface as ui
from . import thresholding
from .models import LogisticModel

import bokeh
from bokeh.models import ColumnDataSource, ColorBar, Range1d, CustomJS, Div, RadioButtonGroup, CDSView, BooleanFilter, TextInput, Button, Circle, \
    HoverTool, RangeSlider, CheckboxGroup
from bokeh.models.widgets import Select, DataTable, TableColumn
from bokeh.transform import linear_cmap
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot, layout
from bokeh import events

# ################################ #
# Define dynamic JS Code Functions #
# ################################ #
#
# Dictionaries are the easiest way to push data between python setup and java script functions later.
# Try to construct logical parameter dicts to manipulate the java objects.
#

def get_js_source_selection_code(cols_ratio):
    js_code = ''.join([
        """
        // Define min max drug variables
        const inds = source.selected.indices;
        var drug_min = Math.min(... doses);
        var drug_max = Math.max(... doses);
        """,

        """
        // Define sigmoid curve
        function sigmoid_curve(x, pec50, slope, front, back) {
            return (front - back) / (1 + Math.pow(10, (slope * (x + pec50)))) + back ;
        };
        """,

        """
        // Define array for sigmoid curve plotting
        function makeDrugArr(startValue, stopValue, cardinality) {
            var arr = [];
            var step = (stopValue - startValue) / (cardinality - 1);
            for (var i = 0; i < cardinality; i++) {
                arr.push(Math.pow(10, startValue + (step * i)));
            }
            return arr;
        };
        """,

        """
        // Define array of length with default fill value
        function fillArr(fill_value, length) {
            var arr = [];
            for (var i = 0; i < length; i++) {
                arr.push(fill_value);
            }
            return arr;
        };
        """,

        # This will display all rows in the table that have been selected
        """
        // Activate Table View of selected indices
        const inds_set = new Set(inds);
        for(var i=0, im=source.data['index'].length; im>i; i++){
            if(inds_set.has(i)){
                TableView.filters[0].booleans[i] = true;
            } else {
                TableView.filters[0].booleans[i] = false;
            };
        };
        """,

        # This part resets all plots to default after deselection
        # The scatter curve plot are set to default values, The fit line will be set to y=1,
        # The y range lim is resetted to (0,2)
        """
        // Deselection shows default plot
        if (inds.length == 0){

            // Plot the data points in scatter plot
            curve_dots.data['x'] = doses;
            curve_dots.data['y'] = default_curve;
            curve_dots.data['labels'] = fillArr('default', doses.length);
            curve_dots.data['names'] = fillArr('example', doses.length);

            // Plot the default sigmoid curve in multi line plot
            curve_fit.data['xs'] = [[drug_min, drug_max]];
            curve_fit.data['ys'] = [[1, 1]];
            curve_fit.data['labels'] = [fillArr('default', 1)];
            curve_fit.data['names'] = [fillArr('example', 1)];

            // Adjust the y range of the curve plot to default values
            CurveFig.y_range.start = 0;
            CurveFig.y_range.end = 2;

            // Set pec50 to NaN in multi line plot
            pec50_line.data['xs'] = [[0, 1]];
            pec50_line.data['ys'] = [[NaN, NaN]];
            pec50_line.data['labels'] = [fillArr('', 1)];
            pec50_line.data['names'] = [fillArr('', 1)];

            // Set Quality to NaN in multi line plot
            quality_line.data['xs'] = [[0, 1]];
            quality_line.data['ys'] = [[NaN, NaN]];
            quality_line.data['labels'] = [fillArr('', 1)];
            quality_line.data['names'] = [fillArr('', 1)];

            // Set Identification to NaN in multi line plot
            identification_line.data['xs'] = [[0, 1]];
            identification_line.data['ys'] = [[NaN, NaN]];
            identification_line.data['labels'] = [fillArr('', 1)];
            identification_line.data['names'] = [fillArr('', 1)];

            // Emit changes
            curve_dots.change.emit();
            curve_fit.change.emit();
            CurveFig.y_range.change.emit();
            CurveFig.change.emit();
            pec50_line.change.emit();
            quality_line.change.emit();
            identification_line.change.emit();
            source.change.emit();
            return;
        };
        """,

        # Enable visualization for all items in the curve plot and in the quality plot
        # Scatter revers to the observed ratios
        # Curve to the curve line plot
        # Quality to the quality plot
        """
        // Enable Viz of all selected elements
        const x_values_scatter = [];
        const y_values_scatter = [];
        const labels_scatter = [];
        const names_scatter = [];

        const curve_x = makeDrugArr(Math.log10(drug_min)-0.5, Math.log10(drug_max)+0.5, 500);
        const x_values_curve = [];
        const y_values_curve = [];
        const labels_curve = [];
        const names_curve = [];

        // Arrays for the histogram lines in pec50, quality, and identification distributions
        const x_values_pec50 = [];
        const y_values_pec50 = [];
        const labels_pec50 = [];
        const names_pec50 = [];

        const x_values_quality = [];
        const y_values_quality = [];
        const labels_quality = [];
        const names_quality = [];

        const x_values_identification = [];
        const y_values_identification = [];
        const labels_identification = [];
        const names_identification = [];
        """,

        """
        for(var i=0, im=inds.length; im>i; i++){

            // Get the data row map for all columns in source data
            var selected_row = {};
            for (const item in source.data) {
                selected_row[item] = source.data[item][inds[i]];
            };

            // Select important values from row
            var idx = inds[i];
            var row_name = selected_row['Name'];
        """,

        # This needs to be auto generated depending on the number of drug doses / TMT columns available to python
        # Unpack to have all the lines joinable in the final list
        '\n'.join([f"    var c{i} = selected_row['{r_col}'];" for i, r_col in enumerate(cols_ratio)]),

        """
            // Extend the scatter arrays for scatter plot
            x_values_scatter.push(... doses);
            y_values_scatter.push(... [""",

        ','.join([f"c{i}" for i, r_col in enumerate(cols_ratio)]),

        """]);
            labels_scatter.push(... fillArr(idx, doses.length));
            names_scatter.push(... fillArr(row_name, doses.length));

            // Calculate the curve y values
            var pec50 = selected_row['pEC50'];
            var slope = selected_row['Curve Slope'];
            var front = selected_row['Curve Front'];
            var back = selected_row['Curve Back'];

            var curve_y = curve_x.map(function(x) {
                return sigmoid_curve(Math.log10(x), pec50, slope, front, back);
            });

            // Append the curves arrays for multi line plot
            x_values_curve.push(curve_x);
            y_values_curve.push(curve_y);
            labels_curve.push([idx]);
            names_curve.push([row_name]);

            // Append the pec50 for multi line plot
            var pec50 = selected_row['pEC50'];
            x_values_pec50.push([0, 1]);
            y_values_pec50.push([pec50, pec50]);
            labels_pec50.push([idx]);
            names_pec50.push([row_name]);

            // Append the signal quality for multi line plot
            var quality = selected_row['Signal Quality'];
            x_values_quality.push([0, 1]);
            y_values_quality.push([quality, quality]);
            labels_quality.push([idx]);
            names_quality.push([row_name]);

            // Append the identification score for multi line plot
            var id_score = selected_row['Score'];
            x_values_identification.push([0, 1]);
            y_values_identification.push([id_score, id_score]);
            labels_identification.push([idx]);
            names_identification.push([row_name]);
        };
        """,

        # Make all the actual updates
        """
        // Plot the data points
        curve_dots.data['x'] = x_values_scatter;
        curve_dots.data['y'] = y_values_scatter;
        curve_dots.data['labels'] = labels_scatter;
        curve_dots.data['names'] = names_scatter;

        // Plot the sigmoid curve
        curve_fit.data['xs'] = x_values_curve;
        curve_fit.data['ys'] = y_values_curve;
        curve_fit.data['labels'] = labels_curve;
        curve_fit.data['names'] = names_curve;

        // Adjust the y range of the plot if there is a bigger value
        //var y_max = Math.max(... y_values_scatter, ... y_values_curve.flat());
        var y_max = Math.max(... y_values_scatter);
        if (y_max > 1.905){
            CurveFig.y_range.end = y_max + y_max * 0.05;
        } else {
            CurveFig.y_range.end = 2;
        };

        // Adjust the distribution lines to the selection
        pec50_line.data['xs'] = x_values_pec50;
        pec50_line.data['ys'] = y_values_pec50;
        pec50_line.data['labels'] = labels_pec50;
        pec50_line.data['names'] = names_pec50;
        quality_line.data['xs'] = x_values_quality;
        quality_line.data['ys'] = y_values_quality;
        quality_line.data['labels'] = labels_quality;
        quality_line.data['names'] = names_quality;
        identification_line.data['xs'] = x_values_identification;
        identification_line.data['ys'] = y_values_identification;
        identification_line.data['labels'] = labels_identification;
        identification_line.data['names'] = names_identification;

        // Emit changes
        curve_fit.change.emit();
        curve_dots.change.emit();
        CurveFig.y_range.change.emit();
        CurveFig.change.emit();
        pec50_line.change.emit();
        quality_line.change.emit();
        identification_line.change.emit();
        source.change.emit();
        """
    ])
    return js_code


def get_js_table_selection_code(col):
    js_code = ''.join([
        """
        // Define Variables
        const new_idxs = [];
        """,

        # Dynamically select the input column
        f"const mod_seq_arr = source.data['{col}'];",

        """
        var user_regex = String(text_input.value);
        var n_rows = source.data['index'].length;
        var selected_curves = selection_view.filters[0].booleans;

        // Prevent to unspecific searches
        if (user_regex.length < 3){
            return;
        };

        // Check each row if substring from user is in mod seq. If so add to new index
        for(var i = 0; i < n_rows; i++){
            if (mod_seq_arr[i].match(user_regex)){
                if (selected_curves[i]){
                    new_idxs.push(i);
                };
            };
        };

        // Update Selection
        source.selected.indices = new_idxs;
        source.change.emit();
        """
    ])
    return js_code


# The JS code is split by conditions with each having a separate for loop,
# because its faster than one loop with checking each conditions. This difference you can feel as user !!
def get_js_visibility_toggle_code():
    js_code = \
        """
        var significant_indicator = source.data['Curve Regulation'];
        var pEC50s = source.data['pEC50'];
        var Scores = source.data['Score'];
        var Signals = source.data['Signal Quality'];
        var n_rows = source.data['index'].length;
        var reg_filter = regulation_toggle.active;
        var pec50_limit = pec50_slider.value;
        var score_limit = score_slider.value;
        var signal_limit = signal_slider.value;
        var toggle_value = true;
        var pEC50_mask = true;
        var score_mask = true;
        var signal_mask = true;

        //console.log("The data set view was changed to: ", reg_filter);
        
        // Function to compute the product of p1 and p2
        function is_within_limits(value, limit) {
            return (value >= limit[0]) && (value <= limit[1]);
        }

        // Select data to show
        if (reg_filter > 0){
            // Deep copy required as it would otherwise overwrite the original values and only work once
            for(var i = 0; i < n_rows; i++){
                pEC50_mask = is_within_limits(pEC50s[i], pec50_limit);
                signal_mask = is_within_limits(Signals[i], signal_limit);
                score_mask = is_within_limits(Scores[i], score_limit);
                toggle_value = (significant_indicator[i] == reg_filter) && pEC50_mask && signal_mask && score_mask;
                view.filters[0].booleans[i] = toggle_value;
            };
        } else {
            for(var i = 0; i < n_rows; i++){
                pEC50_mask = is_within_limits(pEC50s[i], pec50_limit);
                signal_mask = is_within_limits(Signals[i], signal_limit);
                score_mask = is_within_limits(Scores[i], score_limit);
                toggle_value = pEC50_mask && signal_mask && score_mask;
                view.filters[0].booleans[i] = toggle_value;
            };
        };

        // Adjust the threshold lines
        pec50_line1.data['y'] = [pec50_limit[0], pec50_limit[0]];
        pec50_line2.data['y'] = [pec50_limit[1], pec50_limit[1]];
        signal_line1.data['y'] = [signal_limit[0], signal_limit[0]];
        signal_line2.data['y'] = [signal_limit[1], signal_limit[1]];
        score_line1.data['y'] = [score_limit[0], score_limit[0]];
        score_line2.data['y'] = [score_limit[1], score_limit[1]];

        // Emit changes
        pec50_line1.change.emit();
        pec50_line2.change.emit();
        signal_line1.change.emit();
        signal_line2.change.emit();
        score_line1.change.emit();
        score_line2.change.emit();
        source.change.emit();
        """
    return js_code


def get_js_fig1_yaxis_selection():
    js_code = \
        """
            // Take the selection
            var selected_view = view.value;
            //console.log('Change Figure 1 to ' + selected_view + ' View.');
            //console.log(volcano_params);

            // Handle the volcano view with non-corrected p-values
            if (selected_view == 'Volcano'){
                // Define the p-value selection for the volcano plot
                if (p_value_toggle.active == 0) {
                    scatter.glyph.y.field = volcano_params.y_col_name_0;
                    yaxis.axis_label = volcano_params.y_label_0;
                    fig.y_range.end =  volcano_params.y_range_p0[1];
                    fig.y_range.reset_end =  volcano_params.y_range_p0[1];
                    threshold_v0.visible = true;
                    threshold_v1.visible = false;
                } else if (p_value_toggle.active == 1){
                    scatter.glyph.y.field = volcano_params.y_col_name_1;
                    yaxis.axis_label =  volcano_params.y_label_1;
                    fig.y_range.end =  volcano_params.y_range_p1[1];
                    fig.y_range.reset_end =  volcano_params.y_range_p1[1];
                    threshold_v0.visible = false;
                    threshold_v1.visible = true;
                } else {
                    console.log('This p-value toggle value is not implemented: ', p_value_toggle.active);              
                };
                
                // Adjust the x & y axis
                fig.title.text = 'Volcano Plot';
                fig.y_range.start = 0; // actual value
                fig.y_range.reset_start = 0;  // what the reset button will do
                fig.x_range.start = volcano_params.x_range[0];
                fig.x_range.reset_start = volcano_params.x_range[0];
                fig.x_range.end = volcano_params.x_range[1];
                fig.x_range.reset_end = volcano_params.x_range[1];

                // Show the all curves by default (label index 0)
                regulation_toggle.active = 0;
                
                // Hide and show elements
                p_value_toggle.visible = true;
                color_bar.visible = true;
                
                // Hide fold change asymptotes for sam method as there is a continuous threshold line v
                if (volcano_params['method'] == 'sam') {
                    threshold_p.visible = false;
                } else {
                    threshold_p.visible = true;
                };
            };

            // Handle the potency view
            if (selected_view == 'Potency'){
                scatter.glyph.y.field = 'pEC50';
                fig.title.text = 'Potency Plot';
                
                // Adjust the x & y axis
                fig.y_range.start = potency_range[0] - 0.1;
                fig.y_range.reset_start = potency_range[0] - 0.1;
                fig.y_range.end = potency_range[1] + 0.1;
                fig.y_range.reset_end = potency_range[1] + 0.1;
                fig.x_range.start = volcano_params.x_range[0];
                fig.x_range.reset_start = volcano_params.x_range[0];
                fig.x_range.end = volcano_params.x_range[1];
                fig.x_range.reset_end = volcano_params.x_range[1];
                yaxis.axis_label = 'Potency: pEC50';

                // Hide the significant curves only by default (label index 1)
                regulation_toggle.active = 1;
                
                // Hide and show elements
                p_value_toggle.visible = false;
                //color_bar.visible = false;
                threshold_v0.visible = false;
                threshold_v1.visible = false;
                threshold_p.visible = true;
            };

            // Emmit changes
            source.change.emit();

        """
    return js_code


# ####################### #
# Define Python Functions #
# ####################### #


def draw_default_values(n):
    np.random.seed(8)
    s = pd.Series([1, 0.975, 1.025]).sample(n, replace=True)
    return s.values


def get_exponential_limited_space(start, end, inf_limit=0.0, num=5000, exp=3):
    """
    get_exponential_limited_space(start, end, inf_limit=0.0, num=5000, exp=3)

    Make an array from start to end with exponential resolution until the infinity limit.
    Linear resolution between the infinity limits.
    """
    # process input
    inf_limit = abs(inf_limit)
    start = abs(start) - inf_limit
    end = abs(end) - inf_limit

    # Build the x range
    x1 = -np.linspace(start ** (1 / exp), 0, num=num) ** exp - inf_limit
    x2 = np.linspace(-inf_limit, inf_limit, num=num // 10)
    x3 = np.linspace(0, end ** (1 / exp), num=num) ** exp + inf_limit
    x = np.sort(np.unique(np.concatenate((x1, x2, x3))))
    return x


# ################## #
# Dashboard Building #
# ################## #


def dashboard(df, title, out_path, drug_doses, drug_unit, cols_ratio, model, f_statistic_params, bokeh_params, volcano_params, curve_y_range=(0, 2),
              signal_slider_params=(0, 12, 1200, 1200), score_slider_params=(-1, 2, 20, 20),
              pec50_slider_params=(0, 0, 14, 14), plot_scores=True, plot_signal=True):
    """
    Rendering the interactive bokeh dashboard and save it as html at out_path.

    Parameter
    ---------
    df : pd.DataFrame
        Curves files containing all data.
    title : str
        Title of the bokeh file.
    out_path : str
        Path to the output file.
    drug_doses : array-like
        Drug doses
    drug_unit : str
        unit of the drug doses
    cols_ratio : array-like
        All column names of the ratio data.
    f_statistic_params : dict
        parameter dictionary which adjusts the specific statistic and p-value calculations. It must contain at least alpha and fc_lim.
    bokeh_params : dict
        parameter dictionary which adjusts the bokeh rendering options.
    volcano_params: dict
        parameter dictionary which adjusts the volcano plot depending on the selected method and data
    curve_y_range : tuple(<lower_limit>, <upper_limit>)
        default ratio y range of the curve panel.
    signal_slider_params : tuple(<lower_limit>, <lower_default>, <upper_default>, <upper_limit>)
        Slider parameters.
    score_slider_params : tuple(<lower_limit>, <lower_default>, <upper_default>, <upper_limit>)
        Slider parameters.
    pec50_slider_params : tuple(<lower_limit>, <lower_default>, <upper_default>, <upper_limit>)
        Slider parameters.
    plot_scores : True, plot toggle
        Toggle to show or hide the scores histogram plot.
    plot_signal : True, plot toggle
        Toggle to show or hide the signal / quality histogram plot.

    Return
    ------
    None

    File output
    -----------
    interactive html file at the specified path
    """

    #
    # Header DIV
    #

    # Make a dynamic Div that shows the table, empty by default
    header = Div(text=f"<h1>{title}<h1>", width=700, height=65, margin=(5, 5, 5, 5))

    #
    # Data Selection Buttons
    #

    # Volcano Filter
    selection_labels = ["Volcano", "Potency"]
    selection_yaxis = Select(value="Volcano", options=selection_labels, width=100)

    # Volcano p-value selection
    button_labels_pvalue = [volcano_params['button_0'], volcano_params['button_1']]
    button_group_pvalue = RadioButtonGroup(labels=button_labels_pvalue, active=volcano_params['button_default'])

    # Regulation Filter
    button_labels_regulated = ["all", "regulated", "not-regulated"]
    button_group_regulated = RadioButtonGroup(labels=button_labels_regulated, active=0)

    # pEC50 Range Slider for subsetting curves by pEC50
    margin = (10, 20, 10, 10)  # (Top, right, bottom, left)
    pec50_slider = RangeSlider(start=pec50_slider_params[0], end=pec50_slider_params[3], value=pec50_slider_params[1:3],
                               step=.1, title=r"pEC50 Range Selection", margin=margin)

    # Score Range Slider for subsetting curves by search engine score
    slider_steps = min(0.5, abs(score_slider_params[3] - score_slider_params[0]) / 100)
    score_slider = RangeSlider(start=score_slider_params[0], end=score_slider_params[3], value=score_slider_params[1:3],
                               step=slider_steps, title=r"Score Range Selection", margin=margin, visible=plot_scores)

    # Signal Range Slider for subsetting curves by control signal
    slider_steps = min(0.5, abs(signal_slider_params[3] - signal_slider_params[0]) / 100)
    signal_slider = RangeSlider(start=signal_slider_params[0], end=signal_slider_params[3], value=signal_slider_params[1:3],
                                step=slider_steps, title=r"Signal Range Selection", margin=margin, visible=plot_signal)

    # Name Filter
    margin = (10, 5, 30, 5)  # (Top, right, bottom, left)
    visibility = 'Name' in df.columns
    name_input = TextInput(value="", margin=margin, width=300, visible=visibility)
    name_search_button = Button(label="Select Name", button_type="default", margin=margin, width=100, visible=visibility)

    # Mod Sequence Filter
    visibility = 'Modified sequence' in df.columns
    modseq_input = TextInput(value="", margin=margin, width=500, visible=visibility)
    modseq_search_button = Button(label="Select Sequence", button_type="default", margin=margin, width=150, visible=visibility)

    # Gene Filter
    visibility = 'Genes' in df.columns
    gene_input = TextInput(value="", margin=margin, width=300, visible=visibility)
    gene_search_button = Button(label="Select Gene", button_type="default", margin=margin, width=100, visible=visibility)

    #
    # Data
    #

    # All Data in bokeh format
    source = ColumnDataSource(data=df)

    # This default filter should reflect the default option selection based on sliders and buttons at the start
    # The View object allows for interactive subsetting of the data
    view_selected_curves = CDSView(filter=BooleanFilter(np.full(len(df), True).tolist()))
    source_view_table = CDSView(filter=BooleanFilter(np.full(len(df), False).tolist()))

    #
    # Volcano / Potency Plot [1]
    #

    # Color mapper
    viridis_r = tuple(list(bokeh.palettes.Viridis256)[::-1])
    color_mapper = linear_cmap(field_name='pEC50', palette=viridis_r, low=-np.log10(drug_doses.max()), high=-np.log10(drug_doses.min())+1)

    # Calculate the volcano thresholds for standard p-value
    n = (drug_doses > 0).sum() + 1  # 1 for control data point
    dfn, dfd = model.get_dofs(n, optimized=f_statistic_params['optimized_dofs'])
    dfn = f_statistic_params.get('dfn', dfn)
    dfd = f_statistic_params.get('dfd', dfd)
    fc_lim = f_statistic_params['fc_lim']
    alpha = f_statistic_params['alpha']
    two_sided = f_statistic_params['two_sided']
    s0 = thresholding.get_s0(fc_lim=fc_lim, alpha=alpha, dfn=dfn, dfd=dfd, two_sided=two_sided) if f_statistic_params['mtc_method'] == 'sam' else 0.0
    x_cutoff = pd.Series(get_exponential_limited_space(start=2*volcano_params['x_range'][0], end=2*volcano_params['x_range'][1], inf_limit=fc_lim, num=2000))
    y_cutoff = thresholding.map_fc_to_pvalue_cutoff(x_cutoff, alpha=alpha, s0=s0, dfn=dfn, dfd=dfd)
    threshold_v0 = ColumnDataSource({'x': x_cutoff, 'y': y_cutoff})

    # Calculate the volcano thresholds for corrected p-value
    y_cutoff = pd.Series(np.full_like(x_cutoff, -np.log10(alpha)))
    threshold_v1 = ColumnDataSource({'x': x_cutoff, 'y': y_cutoff})

    # Calculate the potency thresholds
    x_cutoff = pd.Series([-fc_lim, -fc_lim, np.nan, fc_lim, fc_lim])
    y_cutoff = pd.Series([-100, 100, np.nan, -100, 100])
    threshold_p = ColumnDataSource({'x': x_cutoff, 'y': y_cutoff})

    # Make the volcano figure (number 1)
    tools = "pan,box_zoom,reset,lasso_select,tap,undo,save"
    fig1 = figure(width=700, height=600, title=volcano_params['title'], tools=tools, toolbar_location="above", output_backend=bokeh_params['backend'])

    # add a circle renderer with a size, color, and alpha
    # By default its a volcano plot view, which can be modulated by JS later by the user.
    # Depending on which button is active, the corresponding y column is rendered initially.
    bd = volcano_params['button_default']
    fig1_dots = fig1.circle(x=volcano_params['x_col_name'], y=volcano_params[f'y_col_name_{bd}'], line_color=color_mapper, color=color_mapper,
                            fill_alpha=0.4, size=6, source=source, view=view_selected_curves)
    fig1_dots.selection_glyph = Circle(line_color='black', fill_color=color_mapper, fill_alpha=1)
    fig1_dots.nonselection_glyph = Circle(line_color=None, fill_color=color_mapper, fill_alpha=0.2)

    # Add hover tooltips labels to figure 1 for dots
    tooltips = [("Dot", "$index, @Name{%.25s}")]
    tooltips_format = {'@Name': 'printf'}
    dots_hover_tool = HoverTool(renderers=[fig1_dots], tooltips=tooltips, formatters=tooltips_format)
    fig1.add_tools(dots_hover_tool)

    # Add thresholds and potency line. Visibility depends on the used approach
    volcano_threshold_line_v0 = fig1.line(x='x', y='y', line_width=1.5, source=threshold_v0, color='crimson', line_dash='solid')
    volcano_threshold_line_v0.visible = True
    volcano_threshold_line_v1 = fig1.line(x='x', y='y', line_width=1.5, source=threshold_v1, color='crimson', line_dash='solid')
    volcano_threshold_line_v1.visible = False
    potency_threshold_line_p = fig1.line(x='x', y='y', line_width=1.5, source=threshold_p, color='crimson', line_dash='solid')
    potency_threshold_line_p.visible = volcano_params['method'] != 'sam'

    # Add color bar
    pec50_color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, title=66 * ' ' + 'pEC50')
    fig1.add_layout(pec50_color_bar, 'right')

    # Set the ranges & labels for figure 1
    fig1.y_range = Range1d(*volcano_params['y_range_p0'])
    fig1.yaxis.axis_label = volcano_params['y_label_0']
    fig1.x_range = Range1d(*volcano_params['x_range'])
    fig1.xaxis.axis_label = volcano_params['x_label']
    fig1.xgrid.visible = False
    fig1.ygrid.visible = False
    fig1.outline_line_color = None

    # Set potency range for alternative fig 1
    potency_range = (pec50_slider_params[0] + 2, pec50_slider_params[3] - 3)

    #
    # Curve Plot [2]
    #

    # Make the curve figure (number 2)
    tools = "save"
    fig2 = figure(width=500, height=600, title="Dose-Response Curve", x_axis_type="log", tools=tools, toolbar_location="above", output_backend=bokeh_params['backend'])

    # Define the Source data
    default_ratios = draw_default_values(len(drug_doses))
    curve_fit_source = ColumnDataSource(data=dict(xs=[[drug_doses.min(), drug_doses.max()]], ys=[[1, 1]], labels=[['default']], names=[['example']]))
    curve_dots_source = ColumnDataSource(data=dict(x=drug_doses, y=default_ratios, labels=len(drug_doses) * ['default'],
                                                   names=len(drug_doses) * ['example']))

    # Plot the Curve plot with fit line and scatter points
    fit_line = fig2.multi_line(xs='xs', ys='ys', color="crimson", line_width=5, alpha=0.6, source=curve_fit_source)
    curve_dots = fig2.circle(x='x', y='y', fill_color='black', fill_alpha=1, source=curve_dots_source, size=7, line_color='black')

    # Add hover tooltips labels to figure 2 for fitted lines and curve dots
    tooltips = [("Curve", "@labels, @names{%.25s}")]
    tooltips_format = {'@names': 'printf'}
    curve_hover_tool = HoverTool(renderers=[fit_line, curve_dots], tooltips=tooltips, formatters=tooltips_format)
    fig2.add_tools(curve_hover_tool)

    # Make figure number 2 pretty
    fig2.y_range = Range1d(*curve_y_range)
    fig2.yaxis.axis_label = 'Relative Response'
    fig2.x_range = Range1d(drug_doses.min() / 10, drug_doses.max() * 10)
    fig2.xaxis.axis_label = f'Concentration [{drug_unit}]'
    fig2.xgrid.visible = False
    fig2.ygrid.visible = False
    fig2.outline_line_color = None

    #
    # Quality Distribution [3]
    #

    # Make the quality distribution figure (number 3)
    tools = "save"
    margin = (0, 5, 0, 5)  # (Top, right, bottom, left)
    fig3 = figure(width=175, height=600, title="Signal", margin=margin, toolbar_location="above", tools=tools, visible=plot_signal)

    # Calculate the data
    quality_array = df.loc[df['Signal Quality'] <= signal_slider_params[3], 'Signal Quality']
    hist, edges = np.histogram(quality_array, density=True, bins=80)
    quality_source = ColumnDataSource(data=dict(xs=[[0, 1]], ys=[[np.nan, np.nan]], labels=[['']], names=[['']]))
    signal_threshold1_source = ColumnDataSource(data=dict(x=[0, 1], y=[signal_slider_params[1], signal_slider_params[1]], labels=['Threshold', 'Threshold']))
    signal_threshold2_source = ColumnDataSource(data=dict(x=[0, 1], y=[signal_slider_params[3], signal_slider_params[3]], labels=['Threshold', 'Threshold']))

    # Plot the data distribution
    hist_boxes_3 = fig3.quad(top=edges[:-1], bottom=edges[1:], left=0, right=hist, fill_color="gray", line_color="white", alpha=1)
    quality_lines = fig3.multi_line(xs='xs', ys='ys', color="crimson", line_width=2.5, alpha=1, source=quality_source)
    threshold1_line = fig3.line(x='x', y='y', color="black", line_width=3, alpha=1, source=signal_threshold1_source, line_dash='dashed')
    threshold2_line = fig3.line(x='x', y='y', color="black", line_width=3, alpha=1, source=signal_threshold2_source, line_dash='dashed')

    # Add hover tooltip to figure 3 for quality lines and threshold lines
    tooltips = [("Line", "@labels, @names{%.25s}")]
    tooltips_format = {'@names': 'printf'}
    quality_hover = HoverTool(renderers=[quality_lines], tooltips=tooltips, formatters=tooltips_format)
    fig3.add_tools(quality_hover)
    tooltips = [("Line", "@labels")]
    quality_hover = HoverTool(renderers=[threshold1_line, threshold2_line], tooltips=tooltips)
    fig3.add_tools(quality_hover)

    # Make figure number 3 pretty
    fig3.y_range = Range1d(signal_slider_params[0], signal_slider_params[3])
    fig3.yaxis.axis_label = 'Log2 Control Signal'
    fig3.x_range = Range1d(0, max(hist[1:]))
    fig3.xaxis.visible = False
    fig3.xgrid.visible = False
    fig3.ygrid.visible = False
    fig3.outline_line_color = None

    #
    # Score Distribution [4]
    #

    # Make the quality distribution figure (number 4)
    tools = "save"
    margin = (0, 5, 0, 5)  # (Top, right, bottom, left)
    fig4 = figure(width=175, height=600, title="Identification", margin=margin, toolbar_location="above", tools=tools, visible=plot_scores)

    # Calculate the data
    identification_array = df['Score'].dropna()
    hist, edges = np.histogram(identification_array, density=True, bins=63)
    identification_source = ColumnDataSource(data=dict(xs=[[0, 1]], ys=[[np.nan, np.nan]], labels=[['']], names=[['']]))
    score_threshold1_source = ColumnDataSource(data=dict(x=[0, 1], y=[score_slider_params[1], score_slider_params[1]], labels=['Threshold', 'Threshold']))
    score_threshold2_source = ColumnDataSource(data=dict(x=[0, 1], y=[score_slider_params[3], score_slider_params[3]], labels=['Threshold', 'Threshold']))

    # Plot the data distribution
    hist_boxes_4 = fig4.quad(top=edges[:-1], bottom=edges[1:], left=0, right=hist, fill_color="gray", line_color="white", alpha=1)
    identification_lines = fig4.multi_line(xs='xs', ys='ys', color="crimson", line_width=2.5, alpha=1, source=identification_source)
    threshold_line1 = fig4.line(x='x', y='y', color="black", line_width=3, alpha=1, source=score_threshold1_source, line_dash='dashed')
    threshold_line2 = fig4.line(x='x', y='y', color="black", line_width=3, alpha=1, source=score_threshold2_source, line_dash='dashed')

    # Add hover tooltips to figure 4 for identification lines and score threshold lines
    tooltips = [("Line", "@labels, @names{%.25s}")]
    tooltips_format = {'@names': 'printf'}
    identification_hover = HoverTool(renderers=[identification_lines], tooltips=tooltips, formatters=tooltips_format)
    fig4.add_tools(identification_hover)
    tooltips = [("Line", "@labels")]
    identification_hover = HoverTool(renderers=[threshold_line1, threshold_line2], tooltips=tooltips)
    fig4.add_tools(identification_hover)

    # Make figure number 4 pretty
    fig4.y_range = Range1d(score_slider_params[0] - 0.1, score_slider_params[3])
    fig4.yaxis.axis_label = 'Search Engine Score'
    fig4.x_range = Range1d(0, max(hist[1:]))
    fig4.xaxis.visible = False
    fig4.xgrid.visible = False
    fig4.ygrid.visible = False
    fig4.outline_line_color = None

    #
    # pEC50 Distribution [5]
    #
    tools = "save"
    margin = (0, 5, 0, 5)  # (Top, right, bottom, left)
    fig5 = figure(width=175, height=600, title="pEC50", margin=margin, toolbar_location="above", tools=tools, visible=True)

    # Calculate the data
    potency_array = df.loc[(df['Curve Regulation'] == 1), 'pEC50'].dropna()
    potency_bins = int(abs(potency_range[1] - potency_range[0]) // 0.1)
    if len(potency_array) > 0:
        hist, edges = np.histogram(potency_array, density=True, bins=np.linspace(potency_range[0], potency_range[1], potency_bins))
        hist = hist / max(hist)
    else:
        # if not a single significant curve is present to guarantee that a nice plot is still drawn with empty background
        hist, edges = [0, 1], [-2, -1, 0]
    potency_source = ColumnDataSource(data=dict(xs=[[0, 1]], ys=[[np.nan, np.nan]], labels=[['']], names=[['']]))
    potency_threshold1_source = ColumnDataSource(data=dict(x=[0, 1], y=[pec50_slider_params[1], pec50_slider_params[1]], labels=['Threshold', 'Threshold']))
    potency_threshold2_source = ColumnDataSource(data=dict(x=[0, 1], y=[pec50_slider_params[3], pec50_slider_params[3]], labels=['Threshold', 'Threshold']))

    # Plot the data distribution
    hist_boxes_5 = fig5.quad(top=edges[:-1], bottom=edges[1:], left=0, right=hist, fill_color="gray", line_color="white", alpha=1)
    potency_lines = fig5.multi_line(xs='xs', ys='ys', color="crimson", line_width=2.5, alpha=1, source=potency_source)
    threshold_line1 = fig5.line(x='x', y='y', color="black", line_width=3, alpha=1, source=potency_threshold1_source, line_dash='dashed')
    threshold_line2 = fig5.line(x='x', y='y', color="black", line_width=3, alpha=1, source=potency_threshold2_source, line_dash='dashed')

    # Add hover tooltips to figure 5 for potency lines and threshold lines
    tooltips = [("Line", "@labels, @names{%.25s}")]
    tooltips_format = {'@names': 'printf'}
    potency_hover = HoverTool(renderers=[potency_lines], tooltips=tooltips, formatters=tooltips_format)
    fig5.add_tools(potency_hover)
    tooltips = [("Line", "@labels")]
    potency_hover = HoverTool(renderers=[threshold_line1, threshold_line2], tooltips=tooltips)
    fig5.add_tools(potency_hover)

    # Make figure number 4 pretty
    fig5.y_range = Range1d(potency_range[0], potency_range[1])
    fig5.yaxis.axis_label = 'Regulated Curve Potency'
    fig5.x_range = Range1d(0, max(hist[1:]))
    fig5.xaxis.visible = False
    fig5.xgrid.visible = False
    fig5.ygrid.visible = False
    fig5.outline_line_color = None

    #
    # Text DIV
    #

    # Make a dynamic Div that shows the table, empty by default
    selection_title = Div(text="<h3>Curve Selection:<h3>", width=1400, height=25)

    #
    # Table
    #

    # Define all possible table columns and filter for available columns in input data
    table_cols = [
        dict(field='Name', title="Name", width=300),
        dict(field='Modified sequence', title="Modified Sequence", width=300),
        dict(field='Genes', title="Gene(s)", width=200),
        dict(field='Proteins', title="Uniprot ID(s)", width=200),
        dict(field='Peptides', title="N Peptides", width=100),
        dict(field='Type', title="Type", width=75),
        dict(field='Score', title="Score", width=75),
        dict(field='pEC50', title="Curve pEC50 [M]", width=110),
        dict(field='Curve Slope', title="Slope", width=110),
        dict(field=volcano_params['x_col_name'], title=volcano_params['x_table'], width=100),
        dict(field='Curve AUC', title="AUC", width=50),
        dict(field='Curve R2', title="R2", width=50),
        dict(field=volcano_params['y_col_name_1'], title=volcano_params['y_table_1'], width=120),
        dict(field='Signal Quality', title="Signal Quality", width=110),
        dict(field='Imputation Position', title="Imputation", width=110),
    ]

    # Filter for columns that are actually available to the user
    all_cols = df.columns.copy()
    if not plot_scores:
        all_cols = all_cols.drop('Score')
    if not plot_signal:
        all_cols = all_cols.drop('Signal Quality')
    table_cols = [TableColumn(**tc) for tc in table_cols if tc['field'] in all_cols]

    # Create the Table
    table_width = sum([col.width for col in table_cols]) + 40  # needs 40 for index
    data_table = DataTable(source=source, columns=table_cols, width=table_width, view=source_view_table, autosize_mode="none")

    #
    # Dynamic Plot linkage with JS
    #

    # Change the y axis of the scatter plot
    js_code = get_js_fig1_yaxis_selection()
    selection_yaxis.js_on_change('value',
                                 CustomJS(args=dict(source=source,
                                                    view=selection_yaxis,
                                                    fig=fig1,
                                                    yaxis=fig1.yaxis[0],
                                                    scatter=fig1_dots,
                                                    volcano_params=volcano_params,
                                                    threshold_v0=volcano_threshold_line_v0,
                                                    threshold_v1=volcano_threshold_line_v1,
                                                    threshold_p=potency_threshold_line_p,
                                                    color_bar=pec50_color_bar,
                                                    potency_range=potency_range,
                                                    regulation_toggle=button_group_regulated,
                                                    p_value_toggle=button_group_pvalue),
                                          code=js_code))

    button_group_pvalue.js_on_change('active',
                                     CustomJS(args=dict(source=source,
                                                        view=selection_yaxis,
                                                        fig=fig1,
                                                        yaxis=fig1.yaxis[0],
                                                        scatter=fig1_dots,
                                                        volcano_params=volcano_params,
                                                        threshold_v0=volcano_threshold_line_v0,
                                                        threshold_v1=volcano_threshold_line_v1,
                                                        threshold_p=potency_threshold_line_p,
                                                        color_bar=pec50_color_bar,
                                                        potency_range=potency_range,
                                                        regulation_toggle=button_group_regulated,
                                                        p_value_toggle=button_group_pvalue),
                                              code=js_code))

    # Select Data points from the volcano based on modified sequence
    js_code = get_js_table_selection_code('Name')
    name_search_button.js_on_click(CustomJS(args=dict(source=source,
                                                      text_input=name_input,
                                                      selection_view=view_selected_curves),
                                            code=js_code))

    # Select Data points from the volcano based on modified sequence
    js_code = get_js_table_selection_code('Modified sequence')
    modseq_search_button.js_on_click(CustomJS(args=dict(source=source,
                                                        text_input=modseq_input,
                                                        selection_view=view_selected_curves),
                                              code=js_code))

    # Select Data points from the volcano based on modified sequence
    js_code = get_js_table_selection_code('Genes')
    gene_search_button.js_on_click(CustomJS(args=dict(source=source,
                                                      text_input=gene_input,
                                                      selection_view=view_selected_curves),
                                            code=js_code))

    # Filter data for regulated curves when clicking on the radio buttons
    js_code = get_js_visibility_toggle_code()
    button_group_regulated.js_on_change('active',
                                        CustomJS(args=dict(source=source,
                                                           view=view_selected_curves,
                                                           regulation_toggle=button_group_regulated,
                                                           pec50_slider=pec50_slider,
                                                           pec50_line1=potency_threshold1_source,
                                                           pec50_line2=potency_threshold2_source,
                                                           score_slider=score_slider,
                                                           score_line1=score_threshold1_source,
                                                           score_line2=score_threshold2_source,
                                                           signal_slider=signal_slider,
                                                           signal_line1=signal_threshold1_source,
                                                           signal_line2=signal_threshold2_source),
                                                 code=js_code))

    pec50_slider.js_on_change('value_throttled',
                              CustomJS(args=dict(source=source,
                                                 view=view_selected_curves,
                                                 regulation_toggle=button_group_regulated,
                                                 pec50_slider=pec50_slider,
                                                 pec50_line1=potency_threshold1_source,
                                                 pec50_line2=potency_threshold2_source,
                                                 score_slider=score_slider,
                                                 score_line1=score_threshold1_source,
                                                 score_line2=score_threshold2_source,
                                                 signal_slider=signal_slider,
                                                 signal_line1=signal_threshold1_source,
                                                 signal_line2=signal_threshold2_source),
                                       code=js_code))

    score_slider.js_on_change('value_throttled',
                              CustomJS(args=dict(source=source,
                                                 view=view_selected_curves,
                                                 regulation_toggle=button_group_regulated,
                                                 pec50_slider=pec50_slider,
                                                 pec50_line1=potency_threshold1_source,
                                                 pec50_line2=potency_threshold2_source,
                                                 score_slider=score_slider,
                                                 score_line1=score_threshold1_source,
                                                 score_line2=score_threshold2_source,
                                                 signal_slider=signal_slider,
                                                 signal_line1=signal_threshold1_source,
                                                 signal_line2=signal_threshold2_source),
                                       code=js_code))

    signal_slider.js_on_change('value_throttled',
                               CustomJS(args=dict(source=source,
                                                  view=view_selected_curves,
                                                  regulation_toggle=button_group_regulated,
                                                  pec50_slider=pec50_slider,
                                                  pec50_line1=potency_threshold1_source,
                                                  pec50_line2=potency_threshold2_source,
                                                  score_slider=score_slider,
                                                  score_line1=score_threshold1_source,
                                                  score_line2=score_threshold2_source,
                                                  signal_slider=signal_slider,
                                                  signal_line1=signal_threshold1_source,
                                                  signal_line2=signal_threshold2_source),
                                        code=js_code))

    # Update plots when selecting data points from volcano plot
    js_code = get_js_source_selection_code(cols_ratio)
    source.selected.js_on_change('indices',
                                 CustomJS(args=dict(source=source,
                                                    curve_dots=curve_dots_source,
                                                    curve_fit=curve_fit_source,
                                                    pec50_line=potency_source,
                                                    quality_line=quality_source,
                                                    identification_line=identification_source,
                                                    doses=drug_doses,
                                                    default_curve=default_ratios,
                                                    CurveFig=fig2,
                                                    TableView=source_view_table),
                                          code=js_code))

    #
    # Finalize Grid Layout
    #
    grid = layout([[header, button_group_regulated, selection_yaxis, button_group_pvalue],
                   [fig1, fig2, fig5, fig3, fig4],
                   [selection_title],
                   [pec50_slider, score_slider, signal_slider],
                   [name_input, name_search_button, modseq_input, modseq_search_button, gene_input, gene_search_button],
                   [data_table]])
    # output to HTML file
    output_file(f"{out_path}", title=title)
    save(grid)


def render(df, config):
    """
    render(df, config)

    main function. This deals with all the dashboard creation.
    """
    # Render dashboard if out path is given.
    out_path = config['Paths'].get('dashboard')
    if not out_path:
        return None
    ui.message(f" * Rendering interactive dashboard using {config['Dashboard']['backend']} backend ...")

    # Load parameters from toml file
    experiments = np.array(config['Experiment']['experiments'])
    drug_concs = np.array(config['Experiment']['doses'])
    drug_scale = config['Experiment']['dose_scale']
    drug_unit = config['Experiment']['dose_unit']
    control_mask = (drug_concs != 0)
    drug_log_concs = -1 * tool.build_drug_log_concentrations(drug_concs[control_mask], drug_scale)
    drug_concs = drug_concs[control_mask] * drug_scale
    experiments = experiments[control_mask]
    cols_ratio = tool.build_col_names('Ratio {}', experiments)
    title = config['Meta'].get('description', '')
    mtc_method = config['F Statistic']['mtc_method']

    # Build the volcano plot parameters
    volcano_params = {
        'method': mtc_method,
        'title': 'Volcano Plot',
        'x_col_name': 'Curve Fold Change',
        'x_label': 'Log2 Curve Fold Change',
        'x_table': 'Fold Change',
        'y_col_name_0': 'Curve Log P_Value',
        'y_label_0': 'Significance: -log10 p_value',
        'y_tabel_0': 'Significance',
        'button_0': 'p-values'}

    if mtc_method == 'sam':
        volcano_params |= {'y_col_name_1': 'Curve Relevance Score',
                           'y_label_1': 'Relevance Score',
                           'y_table_1': 'Relevance',
                           'button_1': 'Relevance',
                           'button_default': 0}
    else:
        volcano_params |= {'y_col_name_1': 'Curve Log P_Value adjusted',
                           'y_label_1': 'Significance: -log10 p_value adjusted',
                           'y_table_1': 'Significance adjusted',
                           'button_1': 'adjusted p-value',
                           'button_default': 1}

    # Setup the curve fit with default values unless specified in the toml file for correct curve model
    model = LogisticModel(slope=config['Curve Fit'].get('slope'), front=config['Curve Fit'].get('front'), back=config['Curve Fit'].get('back'))

    # Define columns to save in the dashboard
    df_cols = [volcano_params['x_col_name'], 'Curve AUC', volcano_params['y_col_name_0'], volcano_params['y_col_name_1'], 'pEC50', 'Curve Slope',
               'Curve Front', 'Curve Back', 'Curve R2', 'Curve Regulation']

    # Add columns if they are present in the data frame
    optional_cols = {'Name': str, 'Modified sequence': str, 'Genes': str, 'Proteins': str, 'Peptides': int, 'Imputation Position': str}
    optional_cols = {col: col_type for col, col_type in optional_cols.items() if col in df.columns}
    df_cols.extend(optional_cols.keys())

    # Clean optional columns for dashboard
    for col, col_type in optional_cols.items():
        if col_type is str:
            df[col] = df[col].replace(np.nan, "").astype(col_type)
        elif col_type is int:
            df[col] = df[col].replace(np.nan, 0).astype(col_type)
        elif col_type is bool:
            df[col] = df[col].replace(np.nan, False).astype(col_type)

    # Define 'Signal Quality' for visualization and data filtering
    if 'Signal Quality' in df.columns:
        df['Signal Quality'] = df['Signal Quality'].replace([-np.inf, np.inf, 0], np.nan)
        min_signal = tool.rounddown(df['Signal Quality'].min() / 1.1)
        max_signal = tool.roundup(df['Signal Quality'].max())
        plot_signal = True
        if np.isnan(min_signal) or np.isnan(max_signal):
            df['Signal Quality'] = df['Signal Quality'].replace(np.nan, 0)
            min_signal, max_signal = -1, 1
            plot_signal = False
    else:
        df['Signal Quality'] = 0
        min_signal, max_signal = -1, 1
        plot_signal = False
    df_cols.append('Signal Quality')
    df['Signal Quality'] = df['Signal Quality'].clip(lower=min_signal, upper=max_signal)
    signal_slider_params = (min_signal, min_signal, max_signal, max_signal)

    # Define 'Score' for visualization and data filtering
    if 'Score' in df.columns:
        df['Score'] = df['Score'].replace([-np.inf, np.inf], np.nan)
        max_score = tool.roundup(df['Score'].max() * 1.1)
        min_score = 0
        plot_scores = True
    else:
        df['Score'] = 0
        min_score, max_score = -1, 1
        plot_scores = False
    df_cols.append('Score')
    df['Score'] = df['Score'].clip(lower=min_score, upper=max_score)
    score_slider_params = (min_score, min_score, max_score, max_score)

    # Define the pEC50 range dynamically which are the extreme doses +- 3 order of magnitude
    min_pec50 = round(min(drug_log_concs) - 4, 1)
    max_pec50 = round(max(drug_log_concs) + 4, 1)
    df['pEC50'] = df['pEC50'].clip(lower=min_pec50, upper=max_pec50)
    df = df.sort_values(by='pEC50').round(3)
    pec50_slider_params = (min_pec50, min_pec50, max_pec50, max_pec50)

    # Encode regulation to all=0, regulated=1, not=2 for radio button selection
    regulation_map = {'up': 1, 'down': 1, 'not': 2}
    df['Curve Regulation'] = df['Curve Regulation'].apply(lambda x: regulation_map.get(x, 0))

    # Define plot y ranges dynamically or use default
    y_volcano_max = df.loc[(df['Curve Regulation'] == 1), volcano_params['y_col_name_0']].max() + 0.1
    y_volcano_max = y_volcano_max if y_volcano_max > 8 else 8
    volcano_params['y_range_p0'] = (0, y_volcano_max)
    y_volcano_max = df.loc[(df['Curve Regulation'] == 1),  volcano_params['y_col_name_1']].max() + 0.1
    y_volcano_max = y_volcano_max if y_volcano_max > 8 else 8
    volcano_params['y_range_p1'] = (0, y_volcano_max)

    # Define plot x ranges dynamically or use default
    fc_max = df[volcano_params['x_col_name']].abs().max() + 0.1
    fc_max = fc_max if fc_max > 10 else 10
    volcano_params['x_range'] = (-fc_max, fc_max)

    # Render the dashboard
    cols = np.concatenate([df_cols, cols_ratio])
    dashboard(df[cols], title, out_path, drug_concs, drug_unit, cols_ratio, model, config['F Statistic'], config['Dashboard'],
              volcano_params=volcano_params, signal_slider_params=signal_slider_params, score_slider_params=score_slider_params,
              pec50_slider_params=pec50_slider_params, plot_scores=plot_scores, plot_signal=plot_signal)

    ui.message(" * Dashboard successfully rendered.")
