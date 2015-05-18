library(shiny)
library(ggplot2)
library(fpc)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  
  dat<-iris
  TrueLabel<-dat$Species
  dat$Species<-NULL

  output$outPlot <- renderPlot({
    model <<-kmeans(dat,centers=input$k,algorithm=input$alg)   
    PredictLabels<<-model$cluster
    dat_visual = dat[,c(input$xaxis,input$yaxis)]
    names(dat_visual)<-c("V1","V2")
    dat_visual$cluster_id = factor(model$cluster)
    
    centers=as.data.frame(model$centers)[,c(input$xaxis,input$yaxis)]
    names(centers)<-c("V1","V2")
    centers$cluster_id=1:input$k
    
    ggplot(data=dat_visual, aes(x=V1, y=V2, color=cluster_id)) + 
      geom_point(size=3) + 
      geom_point(data=centers, aes(x=V1, y=V2, color=as.factor(cluster_id)),size=6,shape=3)+
      xlab(input$xaxis)+
      ylab(input$yaxis)
    
    
    
  })
  
  output$validPlot<- renderPlot({
    result = cluster.stats(dist(dat[,1:4]),model$cluster,silhouette=T,sepindex=T,alt.clustering=as.integer(TrueLabel)*input$k)
    result = as.data.frame(cbind(c(result$corrected.rand,result$avg.silwidth,result$sindex),c("Rand","Silhouette","Seperation Index")))
    result$V1 = as.numeric(as.character(result$V1))
    
    ggplot(data=result,aes(x=V2,y=V1,fill=V2))+geom_bar(stat="identity")+ylim(0,1)+ylab("")+xlab("")+theme(legend.position="none")
    
  })
  

  
})