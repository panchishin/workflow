var imageWorkflow = angular.module('imageWorkflow' , []);


imageWorkflow.controller('mainController', function ($scope,$http,$timeout,$interval) {

    $scope.layer_display = [];
    $scope.data = {
    	'images' : [],
        'training_sessions' : [0,0,0,0,0,0,0],
        'difference_images' : [],
        'sele' : 1,
        'prev' : 2
    };
    $scope.similar_images = [];
    $scope.label_list = {};
    $scope.groups = {};
    $scope.isLabeled = 0;
    $scope.search_order = "forward";
    $scope.search_index = .5;

    function retrieveSnapShot() {
        var snap_shots = localStorage.getItem("snap_shots");
        snap_shots = snap_shots ? snap_shots : "{}";
        snap_shots = JSON.parse(snap_shots);
        return snap_shots;
    }

    $scope.list_snap_shots = function() {
        return Object.keys(retrieveSnapShot()).sort();
    }

    var importantData = 'label_list,groups'.split(",")

    $scope.load_snap_shot = function(snap_shot_name) {
        var snap_shots = retrieveSnapShot();
        snap_shot = snap_shots[snap_shot_name]
        importantData.forEach( function(item) {
            $scope[item] = snap_shot[item];
        })
    }
    
    $scope.remove_snap_shot = function(snap_shot_name) {
        var snap_shots = retrieveSnapShot();
        delete snap_shots[snap_shot_name]
        localStorage.setItem("snap_shots",JSON.stringify(snap_shots))
    }

    $scope.create_snap_shot = function(snap_shot_name) {
        var snap_shots = retrieveSnapShot();
        snap_shots[snap_shot_name] = {};
        importantData.forEach( function(item) {
            snap_shots[snap_shot_name][item] = $scope[item]
        })
        localStorage.setItem("snap_shots",JSON.stringify(snap_shots))
    }

    function getRandomImageIndex() {
        return Math.floor( Math.random() * 55000 )
    }

    $scope.randomizeImage = function() {
        $scope.data.images = []
        $scope.similar_images = []
        for (index=0 ; index < 10 ; index++ ) {
            var image = getRandomImageIndex()
            $scope.data.images.push( image )
            $scope.similar_images.push( { 'id' : image , 'state' : 0 } )
        }
    }
    $scope.randomizeImage()

    $scope.reverseSimilarImages = function() {
        Object.keys($scope.similar_images).forEach( function(key) {
            $scope.similar_images[key].state = 1-$scope.similar_images[key].state
        })
    }

    $scope.reset_session = function() {
        $http({method:"GET" , url : "/reset_session" , cache: false}).then(function successCallback(result) {
            console.log("reset_session");
            $scope.randomizeImage();
        })
    }

    $scope.restore_session = function() {
        $http({method:"GET" , url : "/restore_session" , cache: false}).then(function successCallback(result) {
            console.log("restore_session");
            $scope.randomizeImage();
        })
    }

    $scope.save_session = function() {
        $http({method:"GET" , url : "/save_session" , cache: false}).then(function successCallback(result) {
            console.log("save_session");
        })
    }

    $scope.similar = function(index) {
        $scope.new_label="";
        $http({method:"GET" , url : "/similar/"+index , cache: false}).then(function successCallback(result) {
            $scope.similar_images = []
            for ( var index in result.data.response ) {
              $scope.similar_images.push( { 'id' : result.data.response[index] , 'state' : 1 } );
            }
        })
    }

    $scope.learn = function(index) {
        $http({ method : "GET" , url : "/learn/"+index , cache: false}).then(function successCallback(result) {
            $scope.data.training_sessions[index] += 1;
            $scope.randomizeImage();
        })
    }

    $scope.label_score = function(label) {
        var pos = 0;
        for( var index in $scope.label_list[label] ) {
            if ( $scope.label_list[label][index] ) {
                pos += 1;
            }
        }
        return pos
    }

    $scope.add_to_label = function(label) {
        if ( $scope.label_list[label] == undefined ) {
            $scope.label_list[label] = { };
        }
        var number_of_images = $scope.similar_images.length
        var remaining_list = []
        for ( var index in $scope.similar_images ) {
            var data = $scope.similar_images[index];
            $scope.label_list[label][data.id] = data.state;
            if (! data.state ) {
                remaining_list.push(data)
            }
        }
        $scope.similar_images = remaining_list;
        if ( $scope.currentGroup ) {
            $scope.add_to_group($scope.currentGroup,label)
        }
        $scope.new_label = "";
    }

    $scope.label_predict = function(label) {
        $scope.currentGroup = false
        neg_list = [];
        pos_list = [];
        for( var index in $scope.label_list[label] ) {
            if ( $scope.label_list[label][index] == 0 ) {
                neg_list.push(parseInt(index));
            } else {
                pos_list.push(parseInt(index));
            }
        }

        $http({method:"POST" , url : "/group_predict/0" , cache: false , data:{'order':$scope.search_order,'index':$scope.search_index,'isLabeled':$scope.isLabeled,'grouping':[pos_list,neg_list]}}).then(function successCallback(result) {
            $scope.similar_images = []
            for ( var index in result.data.response.positive ) {
                $scope.similar_images.push( { 'id' : result.data.response.positive[index] , 'state' : 1 } );
            }
            for ( var index in result.data.response.negative ) {
                $scope.similar_images.push( { 'id' : result.data.response.negative[index] , 'state' : 0 } );
            }
        })

        $scope.new_label = label;
    }

    $scope.labelsNotInGroup = function(groupName) {
        return Object.keys($scope.label_list).filter( function(name) { return !$scope.groups[groupName]['labels_in'][name]  } )
    }

    $scope.add_to_group = function(groupName,label) {
        $scope.groups[groupName]['labels_in'][label] = 1
    }

    $scope.group_count_total = function(groupName) {
        var total = 0;
        for ( var label in $scope.groups[groupName]['labels_in'] ) {
            total +=  $scope.label_score(label)
        }
        return total
    }

    $scope.remove_from_group = function(groupName,label) {
        delete $scope.groups[groupName]['labels_in'][label]
    }

    $scope.create_new_group = function(new_group_name) {
        $scope.groups[new_group_name] = { 'labels_in' : { } } ;
    }

    $scope.delete_group = function(group_name) {
        delete($scope.groups[group_name]);
    }

    $scope.previous_data_list = null;
    $scope.pause_training = false;

    $scope.group_predict = function(groupName,label) {
        $scope.currentGroup = groupName
        var data_list = [];
        var labels = Object.keys( $scope.groups[groupName]['labels_in'] )
        if ($scope.pause_training && $scope.previous_data_list) {
            data_list = $scope.previous_data_list
        } else {
            for( var label_index in labels ) {
                var sub_list = [];
                data_list.push(sub_list);
                var label_list = $scope.label_list[labels[label_index]]
                for( var index in label_list ) {
                    if ( label_list[index] == 1 ) {
                        sub_list.push(parseInt(index));
                    }
                }  
            }
            $scope.previous_data_list = data_list
        }
        $http({method:"POST" , url : "/group_predict/" + labels.indexOf(label) , cache: false , data:{'order':$scope.search_order,'index':$scope.search_index,'isLabeled':$scope.isLabeled,'grouping':data_list}  }).then(function successCallback(result) {
            $scope.similar_images = []
            for ( var index in result.data.response.positive ) {
                $scope.similar_images.push( { 'id' : result.data.response.positive[index] , 'state' : (label != -1) } );
            }
            if ( label != -1 ) {
                for ( var index in result.data.response.negative ) {
                    $scope.similar_images.push( { 'id' : result.data.response.negative[index] , 'state' : 0 } );
                }
            }
        })
        if ( label >= 0 ) {
          $scope.new_label = label;
        } else {
          $scope.new_label = "";
        }
    }

    $scope.button_confidence = function(label) {
        return $scope.label_score(label) < 50 ? 'btn-danger' : $scope.label_score(label) < 200 ? 'btn-warning' : 'btn-success'
    }

    /***************************   TSNE CODE  BEGIN   ***********************************/


    var img_size = 16;
    var svg_width = 800;
    var svg_height = 400;

    window.states = [
        { x : .43, y : .67, id : 53303 },
        { x : .140, y : .150, id : 2838 },
        { x : .200, y : .250, id : 43799 },
        { x : .300, y : .120, id : 43798 },
        { x : .50, y : .250, id : 43797 },
        { x : .90, y : .170, id : 43796 }
    ]

    window.svg = d3.select("svg")
    .attr("width", svg_width)
    .attr("height", svg_height);


    var gStates = svg.selectAll("g.state").data( states );

    var gState = gStates.enter().append( "g" )
        .attr({
            "transform" : function( d) {
                return "translate("+ [d.x*svg_width,d.y*svg_height] + ")";
            },
            'class'     : 'state' 
        })
    ;


    gState.append( "rect")
        .attr({
            x : -2 , y : -2 ,
            height : img_size+4 , width : img_size+4 ,
            class   : 'outer'
        })
    ;
    gState.append( "rect")
        .attr({
            height : img_size , width : img_size ,
            class   : 'inner'
        })
    ;        

    gState.append("image")
        .attr( "xlink:href" , function(d) { return "http://localhost:9090/layer-1/" + d.id + "/0" } )
        .attr( { height:img_size, width:img_size })


    svg
    .on( "mousedown", function() {
        if( !d3.event.ctrlKey) {
            d3.selectAll( 'g.selected').classed( "selected", false);
        }

        var p = d3.mouse( this);

        svg.append( "rect")
        .attr({
            rx      : 6,
            ry      : 6,
            class   : "selection",
            x       : p[0],
            y       : p[1],
            width   : 0,
            height  : 0
        })
    })
    .on( "mousemove", function() {
        var s = svg.select( "rect.selection");

        if( !s.empty()) {
            var p = d3.mouse( this),
                d = {
                    x       : parseInt( s.attr( "x"), 10),
                    y       : parseInt( s.attr( "y"), 10),
                    width   : parseInt( s.attr( "width"), 10),
                    height  : parseInt( s.attr( "height"), 10)
                },
                move = {
                    x : p[0] - d.x,
                    y : p[1] - d.y
                }
            ;

            if( move.x < 1 || (move.x*2<d.width)) {
                d.x = p[0];
                d.width -= move.x;
            } else {
                d.width = move.x;       
            }

            if( move.y < 1 || (move.y*2<d.height)) {
                d.y = p[1];
                d.height -= move.y;
            } else {
                d.height = move.y;       
            }
           
            s.attr( d);

                // deselect all temporary selected state objects
            d3.selectAll( 'g.state.selection.selected').classed( "selected", false);

            d3.selectAll( 'g.state >rect.inner').each( function( state_data, i) {
                if( 
                    !d3.select( this).classed( "selected") && 
                        // inner rect inside selection frame
                    state_data.x*svg_width>=d.x && state_data.x*svg_width+img_size<=d.x+d.width && 
                    state_data.y*svg_height>=d.y && state_data.y*svg_height+img_size<=d.y+d.height
                ) {

                    d3.select( this.parentNode)
                    .classed( "selection", true)
                    .classed( "selected", true);
                }
            });
        }
    })
    .on( "mouseup",  function() {
           // remove selection frame
        svg.selectAll( "rect.selection").remove();

            // remove temporary selection marker class
        d3.selectAll( 'g.state.selection').classed( "selection", false);

        $scope.similar_images = [];
        d3.selectAll( 'g.state.selected').each( function(d,count) { $scope.similar_images.push( {'state':1,'id':d.id } ) } )
    })
    .on( "mouseout", function() {
        try {
        if( d3.event.relatedTarget.tagName=='HTML') {
                // remove selection frame
            svg.selectAll( "rect.selection").remove();

                // remove temporary selection marker class
            d3.selectAll( 'g.state.selection').classed( "selection", false);
        } } catch (e) { }
    });

    /***************************   TSNE CODE  END   ***********************************/


});