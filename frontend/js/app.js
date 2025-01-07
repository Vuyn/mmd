<script src="https://d3js.org/d3.v5.min.js"></script>

d3.csv("movies.csv").then(function (data) {
    var movies = data;
    var button = d3.select("#button");
    var form = d3.select("#form");
    button.on("click", runEnter);
    form.on("submit", runEnter);
});