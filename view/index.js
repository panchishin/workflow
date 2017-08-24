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
    $scope.confidence = "high";

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

        $http({method:"POST" , url : "/group_predict/0" , cache: false , data:{'confidence':$scope.confidence,'grouping':[pos_list,neg_list]}}).then(function successCallback(result) {
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
        $http({method:"POST" , url : "/group_predict/" + labels.indexOf(label) , cache: false , data:{'confidence':$scope.confidence,'grouping':data_list}  }).then(function successCallback(result) {
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
});