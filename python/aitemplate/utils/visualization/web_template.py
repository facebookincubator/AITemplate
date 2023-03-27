#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Web template for visualization
"""

# flake8: noqa

import jinja2

INDEX_TEMPLATE = jinja2.Template(
    """
<!DOCTYPE html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
  <title>{{network_name}}</title>
</head>

<style>

html {
  scroll-behavior: smooth;
}

* { box-sizing: border-box; }
body {
  font: 16px Arial;
}
.autocomplete {
  /*the container must be positioned relative:*/
  position: relative;
  display: inline-block;
}
input {
  border: 1px solid transparent;
  background-color: #f1f1f1;
  padding: 10px;
  font-size: 16px;
}
input[type=text] {
  background-color: #f1f1f1;
  width: 100%;
}
input[type=submit] {
  background-color: DodgerBlue;
  color: #fff;
}
.autocomplete-items {
  position: absolute;
  border: 1px solid #d4d4d4;
  border-bottom: none;
  border-top: none;
  z-index: 99;
  /*position the autocomplete items to be the same width as the container:*/
  top: 100%;
  left: 0;
  right: 0;
}
.autocomplete-items div {
  padding: 10px;
  cursor: pointer;
  background-color: #fff;
  border-bottom: 1px solid #d4d4d4;
}
.autocomplete-items div:hover {
  /*when hovering an item:*/
  background-color: #e9e9e9;
}
.autocomplete-active {
  /*when navigating through the items using the arrow keys:*/
  background-color: DodgerBlue !important;
  color: #ffffff;
}

.popover {
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  title-bg: "#0d6efd";
}

.header {
  position: fixed;
  width: 100%;
  top: 0;
  left: 0;

}

</style>


<body>

<nav id="nav_bar" class="navbar fixed-top bg-light">
  <div class="container-fluid">
    <a onclick="back_to_head()" class="navbar-brand">{{network_name}}</a>
    <div class="navbar-right">
        <div class="autocomplete" style="width:300px;">
        <input id="name_input" class="form-control me-2" type="search" placeholder="Search" aria-label="Search">
        </div>
        <button class="btn btn-outline-success" onclick="launch_modal_with_input()">Search</button>
    </div>
  </div>
</nav>


  <script
  src="https://code.jquery.com/jquery-3.6.0.js"
  integrity="sha256-H+K7U5CnXl1h5ywQfKtSj8PCmoN9aaq30gDh27Xc0jk="
  crossorigin="anonymous"></script>
  
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
  <script src="https://d3js.org/d3.v5.min.js"></script>
  <script src="https://unpkg.com/@hpcc-js/wasm@0.3.11/dist/index.min.js"></script>
  <script src="https://unpkg.com/d3-graphviz@3.0.5/build/d3-graphviz.js"></script>



  <div id="graph" style="text-align: center;"></div>
  {{modals}}
  

  <script>
  items = {{items}};
  function autocomplete(inp, arr) {
  /*the autocomplete function takes two arguments,
  the text field element and an array of possible autocompleted values:*/
  var currentFocus;
  /*execute a function when someone writes in the text field:*/
  inp.addEventListener("input", function(e) {
      var a, b, i, val = this.value;
      /*close any already open lists of autocompleted values*/
      closeAllLists();
      if (!val) { return false;}
      currentFocus = -1;
      /*create a DIV element that will contain the items (values):*/
      a = document.createElement("DIV");
      a.setAttribute("id", this.id + "autocomplete-list");
      a.setAttribute("class", "autocomplete-items");
      /*append the DIV element as a child of the autocomplete container:*/
      this.parentNode.appendChild(a);
      /*for each item in the array...*/
      for (i = 0; i < arr.length; i++) {
        /*check if the item starts with the same letters as the text field value:*/
        if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
          /*create a DIV element for each matching element:*/
          b = document.createElement("DIV");
          /*make the matching letters bold:*/
          b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
          b.innerHTML += arr[i].substr(val.length);
          /*insert an input field that will hold the current array item's value:*/
          b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
          /*execute a function when someone clicks on the item value (DIV element):*/
              b.addEventListener("click", function(e) {
              /*insert the value for the autocomplete text field:*/
              inp.value = this.getElementsByTagName("input")[0].value;
              /*close the list of autocompleted values,
              (or any other open lists of autocompleted values:*/
              closeAllLists();
          });
          a.appendChild(b);
        }
      }
  });
  /*execute a function presses a key on the keyboard:*/
  inp.addEventListener("keydown", function(e) {
      var x = document.getElementById(this.id + "autocomplete-list");
      if (x) x = x.getElementsByTagName("div");
      if (e.keyCode == 40) {
        /*If the arrow DOWN key is pressed,
        increase the currentFocus variable:*/
        currentFocus++;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 38) { //up
        /*If the arrow UP key is pressed,
        decrease the currentFocus variable:*/
        currentFocus--;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 13) {
        /*If the ENTER key is pressed, prevent the form from being submitted,*/
        e.preventDefault();
        if (currentFocus > -1) {
          /*and simulate a click on the "active" item:*/
          if (x) x[currentFocus].click();
        }
      }
  });
  function addActive(x) {
    /*a function to classify an item as "active":*/
    if (!x) return false;
    /*start by removing the "active" class on all items:*/
    removeActive(x);
    if (currentFocus >= x.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = (x.length - 1);
    /*add class "autocomplete-active":*/
    x[currentFocus].classList.add("autocomplete-active");
  }
  function removeActive(x) {
    /*a function to remove the "active" class from all autocomplete items:*/
    for (var i = 0; i < x.length; i++) {
      x[i].classList.remove("autocomplete-active");
    }
  }
  function closeAllLists(elmnt) {
    /*close all autocomplete lists in the document,
    except the one passed as an argument:*/
    var x = document.getElementsByClassName("autocomplete-items");
    for (var i = 0; i < x.length; i++) {
      if (elmnt != x[i] && elmnt != inp) {
      x[i].parentNode.removeChild(x[i]);
    }
  }
}
/*execute a function when someone clicks in the document:*/
document.addEventListener("click", function (e) {
    closeAllLists(e.target);
});
}
  autocomplete(document.getElementById("name_input"), items);

  function back_to_head() {
    var modal_id = items[0];
    var modal = document.getElementById(modal_id);
    modal.scrollIntoView({ block: 'center',  behavior: 'smooth' });
  }

  function launch_modal_with_input() {
    var modal_id = document.getElementById("name_input").value;
    var modal = document.getElementById(modal_id);
    if (modal == null) {
        var msg = "Could not find node with name: " + modal_id;
        alert(msg);
    } else {
        modal.scrollIntoView({ block: 'center',  behavior: 'smooth' });
        var obj = $("#" + modal_id);
        var shape = obj.find("polygon:first");
        var color = shape.attr("stroke");
        shape.attr("fill", color);
        for (let i = 0; i < 5; i++) {
            obj.fadeOut(100).fadeIn(100).fadeOut(100).fadeIn(100);
        }
        obj.promise().done(function(){
            shape.attr("fill", "none");
        });
    }
  }




  </script>




  <script>
    var dotSrc = `{{dot_src}}`;
    var popover_data = {{popover_data}};
    var graphviz = d3.select("#graph").graphviz();
    var pop_finish = 0;
    // var dotSrcLines;

    function add_popover() {
      for (let i = 0; i < items.length; i++) {
        var id = items[i];
        var obj = $("#" + id);
        obj.attr("data-content", popover_data[id]);
        obj.attr("rel", "popover");
        obj.attr("data-original-title", id);
        obj.attr("data-placement", "top");
        obj.attr("data-trigger", "hover");
        obj.popover();
      }
    }
  

    function render() {
      // console.log('DOT source =', dotSrc);
      // dotSrcLines = dotSrc.split('\\n');
      graphviz.transition(function() {
        return d3.transition().delay(100).duration(1000);
      }).renderDot(dotSrc).on("end", interactive);
    }

    function launch_modal(modal_id) {
      $('#' + modal_id + "_modal").modal('show');
    }


    function interactive() {
      nodes = d3.selectAll('.node,.edge');
      nodes.on("click", function() {
        var id = d3.select(this).attr('id');
        // console.log('Element id="%s"', id);
        document.getElementById("name_input").value = id;
        launch_modal(id);
      });
      nodes.on("mouseover", function() {
        if (pop_finish == 0) {
            add_popover();
            pop_finish = 1;
        }
        var id = d3.select(this).attr("id");
        // console.log('Move over Element id="%s"', id);
        var obj = $("#" + id);
        var shape = obj.find("polygon:first");
        var color = shape.attr("stroke");
        shape.attr("fill", color);
        
      });
      nodes.on("mouseout", function() {
        var id = d3.select(this).attr("id");
        // console.log('Move off Element id="%s"', id);
        var obj = $("#" + id);
        var shape = obj.find("polygon:first");
        shape.attr("fill", "none");
      });

    }
    render(dotSrc);
  </script>
  
</body>
"""
)


MODAL_TEMPLATE = jinja2.Template(
    """
<div class="modal fade" id="{{modal_id}}" tabindex="-1" role="dialog" aria-labelledby="{{modal_label}}" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="{{modal_label}}">{{modal_title}}</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        {{modal_content}}
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-primary" data-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>
"""
)
TABLE_TEMPLATE = jinja2.Template(
    """
<table class="table">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Attributes</th>
      <th scope="col">Value</th>
    </tr>
  </thead>
  <tbody> {% for key, value in table_data.items() %} <tr>
      <td> {{key}} </td>
      <td> {{value}} </td>
    </tr> {% endfor %} </tbody>
</table>
"""
)
