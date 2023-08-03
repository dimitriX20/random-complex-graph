var width = window.innerWidth,
    height = window.innerHeight;

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);

    var squareSize = 2400;
    var xOffset = (width - squareSize) / 2;
    var yOffset = (height - squareSize) / 2;

d3.json("http://localhost:5000/graph").then(function(graph) { 
    console.log(`            nodes: ${graph.nodes.length}, edges: ${graph.edges.length}, γ: ${graph.gamma}, α: ${graph.alpha}`);
    console.log(`            #components: ${graph.amountComp}, size of largest component: ${graph.largestComponentSz}`);
    console.log(`            diameter: ${graph.diameter}, maximum edge distance: ${graph.mxEdge}`);
    console.log(`            average distance: ${graph.avgDist}`);
    console.log(`            clustering coeffcient: ${graph.clusteringCoef}`);

    var rect = svg.append("rect")
        .attr("x", 150)
        .attr("y", 150)
        .attr("width", width - 300)
        .attr("height", height - 300)
        .style("stroke", "black")
        .style("fill", "none");

    var edge = svg.append("g")
        .attr("class", "edges")
        .selectAll("line")
        .data(graph.edges)
        .enter().append("line")
        .attr("x1", function(d) { return xOffset + graph.nodes[d.source].x * squareSize; })
        .attr("y1", function(d) { return yOffset + graph.nodes[d.source].y * squareSize; })
        .attr("x2", function(d) { return xOffset + graph.nodes[d.target].x * squareSize; })
        .attr("y2", function(d) { return yOffset + graph.nodes[d.target].y * squareSize; })
        .style("stroke", function(d) { return d.is_diam_path ? "red" : "black"; })
        .style("stroke-width", function(d) { return d.is_diam_path ? "7px" : "1.7px"; }); // "enhance" edge width 
    

    var node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("r", function(d) { return (0.3 + Math.log2(d.weight)) * 22; })
        .style("fill", "blue")
        .attr("cx", function(d) { return xOffset + d.x * squareSize; })
        .attr("cy", function(d) { return yOffset + d.y * squareSize; })

    function ticked() {
        edge
            .attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });
    
        node
            .attr("cx", function(d) { 
                var radius = d.weight * 15;
                return d.x = Math.max(radius, Math.min(width - radius, d.x)); 
            })
            .attr("cy", function(d) { 
                var radius = d.weight * 15;
                return d.y = Math.max(radius, Math.min(height - radius, d.y)); 
            });
    }
});