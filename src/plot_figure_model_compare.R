library(tidyverse)
library(patchwork)

# Create data frame
df <- data.frame(
  spec = rep(c("nohist", "prevresp_v", "prevresp_z", "prevresp_zv"), each = 3),
  model = rep(c("angle", "ddm", "weibull"), times = 4),
  LOO = c(
    # nohist
    -9008289.299645187, -8994023.291794986, -9047660.295791518,
    # prevresp_v
    -9059819.21880516, -9050140.548590273, -9099162.0289663,
    # prevresp_z
    -8988470.952673083, -9054713.515878558, -9075964.868906405,
    # prevresp_zv
    -9032349.777266849, -9046704.42705331, -9098131.9082945),
  WAIC = c(
    # nohist
    -41953430.49220886, -44511874.30986344, -33147325.72487501,
    # prevresp_v
    -48018504.17537323, -45449736.54303465, -34219706.664712444,
    # prevresp_z
    -46226678.54398163, -44207308.92160495, -30663860.512938887,
    # prevresp_zv
    -46819350.26095487, -45452148.402307145, -35169825.31609502)
)

# Function to compute differences for one metric
compute_diffs <- function(data, metric) {
  result <- data.frame()
  
  for(s in unique(data$spec)) {
    vals <- data[data$spec == s, metric]
    
    result <- rbind(result, data.frame(
      spec = s,
      comparison = c("angle-ddm", "angle-weibull", "ddm-weibull"),
      diff = c(
        vals[1] - vals[2],  # angle - ddm
        vals[1] - vals[3],  # angle - weibull
        vals[2] - vals[3]   # ddm - weibull
      )
    ))
  }
  
  return(result)
}

# Compute differences
loo_diffs <- compute_diffs(df, "LOO")
waic_diffs <- compute_diffs(df, "WAIC")

# Create plot function
plot_diffs <- function(diff_data, title) {
  max_abs <- max(abs(diff_data$diff))
  
  ggplot(diff_data, aes(x = comparison, y = spec)) +
    geom_tile(aes(fill = diff), color = "white") +
    geom_text(aes(label = sprintf("%.0f", diff)), size = 3) +
    scale_fill_gradient2(
      low = "#4575B4", mid = "white", high = "#D73027",
      midpoint = 0, limits = c(-max_abs, max_abs)
    ) +
    scale_y_discrete(limits = rev(c("nohist", "prevresp_v", "prevresp_z", "prevresp_zv"))) +
    theme_minimal() +
    labs(title = title, fill = "Difference") +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
}

# Create plots
p1 <- plot_diffs(loo_diffs, "LOO")
p2 <- plot_diffs(waic_diffs, "WAIC")

# Combine plots
p1 + p2 + 
  plot_layout(guides = "collect") +
  plot_annotation(title = "Model Comparison Differences")

df %>%
  mutate(model_spec = paste(model, spec, sep="_")) %>%
  arrange(LOO) %>%
  select(model_spec, LOO, WAIC)