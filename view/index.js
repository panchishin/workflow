var imageWorkflow = angular.module('imageWorkflow' , []);


imageWorkflow.controller('mainController', function ($scope,$http,$timeout,$interval) {

    function initialize() {
        $scope.data = {
        	'images' : [],
            'similar_embed' : []
        };
        $scope.similar_images = [];
        $scope.error_data = [];
        $scope.label_list = {};
        $scope.groups = {};
        $scope.isLabeled = 0;
        $scope.search_order = "forward";
        $scope.search_index = .5;
        $scope.data_size = 0;
        $scope.dataset_name = "None";
        $scope.layer_display=[];
    }
    initialize();

    $scope.label_listing = function(){
        var data = $scope.label_list ?$scope.label_list : {}
        var result = Object.keys(data);
        results.sort();
        return result;
    }

    function retrieveSnapShot() {
        var snap_shots = localStorage.getItem("snap_shots");
        snap_shots = snap_shots ? snap_shots : "{}";
        snap_shots = JSON.parse(snap_shots);
        return snap_shots;
    }

    $scope.list_snap_shots = function() {
        return Object.keys(retrieveSnapShot()).sort();
    }

    var importantData = ['label_list','groups']

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
        return Math.floor( Math.random() *  $scope.data_size )
    }

    $scope.randomizeImage = function() {
        $scope.data.images = [];
        $scope.similar_embed = [[],[],[],[],[],[],[],[],[],[]];
        $scope.similar_images = [];
        for (index=0 ; index < 10 ; index++ ) {
            var image = getRandomImageIndex()
            $scope.data.images.push( image )
            $scope.similar_images.push( { 'id' : image , 'state' : 0 } )
        }
    }



    $scope.showSimilarToEmbed = function() {
        $scope.similar_embed = [[],[],[],[],[],[],[],[],[],[]];
        for (index=0 ; index < 10 ; index++ ) {
            var image = $scope.data.images[index];
            function callback(image,index){
                var _index = index+0;
                var _image = image+1;
                return function(response) {
                    $scope.similar_embed[_index] = response;
                }
            }
            getSimilar(image,callback(image,index))
        }
    }

    $scope.reverseSimilarImages = function() {
        Object.keys($scope.similar_images).forEach( function(key) {
            $scope.similar_images[key].state = 1-$scope.similar_images[key].state
        })
    }

    $scope.session = function(action) {
        $http({method:"GET" , url : "/session/" + action , cache: false}).then(function successCallback(result) {
            console.log("session "+action);
            $scope.randomizeImage();
        })
    }

    function getSimilar(index,callback) {
        $http({method:"GET" , url : "/similar/"+index , cache: false}).then(function successCallback(result) {
            callback(result.data.response)
        })
    }

    $scope.similar = function(index) {
        $scope.new_label="";
        getSimilar(index,function(response){
            $scope.similar_images = []
            for ( var index in response ) {
              $scope.similar_images.push( { 'id' : response[index] , 'state' : 1 } );
            }
        })
    }

    $scope.learn = function(index) {
        $http({ method : "GET" , url : "/learn/"+index , cache: false}).then(function successCallback(result) {
            $scope.randomizeImage();
            var retrain = function() {
                $scope.learn(index);
            }
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
            $scope.error_data = result.data.response.error.map(function(b){ return b.map(function(c) { return (Math.round(c*1000)*.1).toFixed(1) }) })
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
        $scope.similar_images = [];
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
            $scope.error_data = result.data.response.error.map(function(b){ return b.map(function(c) { return (Math.round(c*1000)*.1).toFixed(1) }) })
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

    $scope.scale_tsne = function(amount) {
        $scope.scale_amount_previous = $scope.scale_amount;
        $scope.scale_amount = amount;
        d3.select("div.svg_border svg").selectAll("g.state")
            .attr({
                "transform" : translate_tsne
            })
        ;
    }

    var svg_width = 1000;
    var svg_height = 700;
    $scope.zoom_center_x = 0.5;
    $scope.zoom_center_y = 0.5;
    $scope.scale_amount = 1.0;
    $scope.scale_amount_previous = 1.0;
    function scale_data_for_tsne(a,center,scale,size) {
        return ( ( a - center ) * scale + center ) * size
    }
    function translate_tsne(data) {
        return "translate("+ [ 
            scale_data_for_tsne(data.x,$scope.zoom_center_x,$scope.scale_amount,svg_width),
            scale_data_for_tsne(data.y,$scope.zoom_center_y,$scope.scale_amount,svg_height)
            ] + ")";
    }

    $scope.tsne_creation = function(states) {
        var img_size = 16;

        d3.select("div.svg_border").select("svg").remove();


        var svg = d3.select("div.svg_border").append("svg")
        .attr("width", svg_width)
        .attr("height", svg_height);

        
        var zoom = d3.behavior.zoom()
            .scaleExtent([1, 4])
            .on("zoom", zoomed);

        function zoomed() {
          console.log(d3.event.sourceEvent.type)
          if ( d3.event.sourceEvent.type == "wheel" ) {
              $scope.zoom_center_x = d3.event.sourceEvent.layerX / svg_width * 1.2 - .1;
              $scope.zoom_center_y = d3.event.sourceEvent.layerY / svg_height * 1.2 - .1;
              $scope.scale_tsne(1.0 * d3.event.scale)
            }
        }

        svg.call(zoom);

        var gState = svg.selectAll("g.state").data( states ).enter().append( "g" )
            .attr({
                "transform" : translate_tsne,
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
            .attr( "xlink:href" , function(d) { return "/layer-1/" + d.id } )
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
                    selection_box = {
                        x       : parseInt( s.attr( "x"), 10),
                        y       : parseInt( s.attr( "y"), 10),
                        width   : parseInt( s.attr( "width"), 10),
                        height  : parseInt( s.attr( "height"), 10)
                    },
                    move = {
                        x : p[0] - selection_box.x,
                        y : p[1] - selection_box.y
                    }
                ;

                if( move.x < 1 || (move.x*2<selection_box.width)) {
                    selection_box.x = p[0];
                    selection_box.width -= move.x;
                } else {
                    selection_box.width = move.x;       
                }

                if( move.y < 1 || (move.y*2<selection_box.height)) {
                    selection_box.y = p[1];
                    selection_box.height -= move.y;
                } else {
                    selection_box.height = move.y;       
                }
               
                s.attr(selection_box);

                    // deselect all temporary selected state objects
                d3.selectAll( 'g.state.selection.selected').classed( "selected", false);

                d3.selectAll( 'g.state >rect.inner').each( function( state_data, i) {
                    var scaled_x = scale_data_for_tsne(state_data.x,$scope.zoom_center_x,$scope.scale_amount,svg_width);
                    var scaled_y = scale_data_for_tsne(state_data.y,$scope.zoom_center_y,$scope.scale_amount,svg_height);
                    if( 
                        !d3.select( this).classed( "selected") && 
                        scaled_x>=selection_box.x && scaled_x+img_size<=selection_box.x+selection_box.width && 
                        scaled_y>=selection_box.y && scaled_y+img_size<=selection_box.y+selection_box.height
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
    }

    $scope.get_tsne_batch = function() {
        $http({method:"GET" , url : "/tsne/1000", cache: false}).then(function successCallback(result) {
            let response = result.data.response.map(function(data) {
                return {'x': data.x / 1000.0 , 'y' : data.y / 1000.0 , 'id' : data.id}
            })

            $scope.tsne_creation( response )
        })
    }


    $scope.get_dataset = function() {
        $http({method:"GET" , url : "/dataset" , cache: false}).then(function successCallback(result) {
            $scope.dataset_name = parseInt(result.data.response.value)
            $scope.data_size = parseInt(result.data.response.size)
            $scope.randomizeImage();
        })
    }

    $scope.choose_dataset = function(name) {
        $http({method:"POST" , url : "/dataset" , data : {'value':name}, cache: false}).then(function successCallback(result) {
            $scope.dataset_name = name
            $scope.data_size = parseInt(result.data.response.size)
            $scope.randomizeImage();
        })
    }


    /***************************   TSNE CODE  END   ***********************************/


});