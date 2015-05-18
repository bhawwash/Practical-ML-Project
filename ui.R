library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Clustering Iris Data"),
  sidebarLayout(
    sidebarPanel(width=3,
      h4("Description"),
      p("This app shows the results of using K-Means to cluster Iris Dataset. You can change the number of clusters and/or the type of algorithm. Moreover, you can select which features to show. You can compare the performance using the vlidation metrics"),
      sliderInput("k",label=h4("Number of Clusters:"),
                  min = 2,max = 10,value = 5),
      radioButtons("alg", label = h4("Algorithm Used"),
                   choices = list("Hartigan-Wong" = "Hartigan-Wong",
                                  "Lloyd" = "Lloyd",
                                  "MacQueen" = "MacQueen"),selected = "Lloyd"),
      selectInput("xaxis", label = h4("X-Axis"), 
                  choices = list("Sepal Length" = "Sepal.Length",
                                 "Sepal Width" = "Sepal.Width",
                                 "Petal Length" = "Petal.Length",
                                "Petal Width"="Petal.Width"),selected = "Sepal.Length"),
      selectInput("yaxis", label = h4("Y-Axis"), 
                  choices = list("Sepal Length" = "Sepal.Length",
                                 "Sepal Width" = "Sepal.Width",
                                 "Petal Length" = "Petal.Length",
                                 "Petal Width"="Petal.Width"),selected = "Sepal.Width")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      h3("Clustering Output"),plotOutput("outPlot"),
      h3("Validation Metrics"),plotOutput("validPlot")
    )
  )
))