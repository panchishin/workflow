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
    $scope.label_errors = {};
    $scope.errors = {};
    $scope.groups = {};

    $scope.randomizeImage = function() {
        $scope.data.images = []
        for (index=0 ; index < 10 ; index++ ) {
            $scope.data.images.push( Math.floor( Math.random() * 10000 ) )
        }
    }
    $scope.randomizeImage()

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
        console.log("calling similar ...");
        $http({method:"GET" , url : "/similar/"+index , cache: false}).then(function successCallback(result) {
            console.log("... done");
            $scope.similar_images = []
            for ( var index in result.data.response ) {
              $scope.similar_images.push( { 'id' : result.data.response[index] , 'state' : 1 } );
            }
        })
        $scope.errors = {};
    }

    $scope.learn = function(index) {
        console.log("LEARNING...");
        $http({ method : "GET" , url : "/learn/"+index , cache: false}).then(function successCallback(result) {
            console.log("...DONE");
            $scope.data.training_sessions[index] += 1;
            $scope.randomizeImage();
        })
    }

    $scope.label_score = function(label) {
        var pos = 0;
        var neg = 0;
        for( var index in $scope.label_list[label] ) {
            if ( $scope.label_list[label][index] == 0 ) {
                neg += 1;
            } else {
                pos += 1;
            }
        }
        //return '+' + pos + ":-" + neg + ":E" + Math.round($scope.label_errors[label]*10)
        return pos
    }

    $scope.add_to_label = function(label) {
        if ( $scope.label_list[label] == undefined ) {
            $scope.label_list[label] = { };
        }
        console.log("Add these to label " + label);
        var number_of_images = $scope.similar_images.length
        if ( $scope.label_errors[label] == undefined ) {
            $scope.label_errors[label] = 0;
        }
        for ( var index in $scope.similar_images ) {
            var data = $scope.similar_images[index];
            $scope.label_list[label][data.id] = data.state;
            if ( $scope.errors[data.id] ) {
                $scope.label_errors[label] += 1.0 - index / number_of_images
            }
        }
        $scope.errors = {};
        $scope.similar_images = [];
        if ( $scope.label_score(label) > 0 && $scope.label_score(label) < 50 ) {
            if ( $scope.currentGroup ) {
                $scope.group_predict($scope.currentGroup,label); 
            } else {
                $scope.label_predict(label); 
            }
        } else {
            $scope.new_label = "";
        }
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

        $http({method:"POST" , url : "/group_predict/0" , cache: false , data:[pos_list,neg_list]}).then(function successCallback(result) {
            $scope.similar_images = []
            for ( var index in result.data.response.positive ) {
                $scope.similar_images.push( { 'id' : result.data.response.positive[index] , 'state' : 1 } );
            }
            for ( var index in result.data.response.negative ) {
                $scope.similar_images.push( { 'id' : result.data.response.negative[index] , 'state' : 0 } );
            }
        })

        $scope.new_label = label;
        $scope.errors = {};
    }

    $scope.errorCount = function() {
        return Object.keys($scope.errors).length;
    }

    $scope.toggleError = function(id) {
        if ( $scope.errors[id] ) {
            delete $scope.errors[id] ;
        } else {
            $scope.errors[id] = 1;
        }
        console.log("We toggled ",id," to be",$scope.errors[id]," bringing the total errors to ", $scope.errorCount() );
    }

    $scope.labelsNotInGroup = function(groupName) {
        console.log("CALLING labelsNotInGroup with "+groupName);
        return Object.keys($scope.label_list).filter( function(name) { console.log("iterating with "+name ); return !$scope.groups[groupName]['labels_in'][name]  } )
    }

    $scope.add_to_group = function(groupName,label) {
        $scope.groups[groupName]['labels_in'][label] = 1
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

    $scope.group_predict = function(groupName,label) {
        $scope.currentGroup = groupName
        var data_list = [];
        var labels = Object.keys( $scope.groups[groupName]['labels_in'] )
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
        $http({method:"POST" , url : "/group_predict/" + labels.indexOf(label) , cache: false , data:data_list}).then(function successCallback(result) {
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
        $scope.errors = {};
    }

    $scope.button_confidence = function(label) {
        return $scope.label_score(label) < 10 ? 'btn-danger' : $scope.label_score(label) < 50 ? 'btn-warning' : 'btn-success'
    }
});