var imageWorkflow = angular.module('imageWorkflow' , []);


imageWorkflow.controller('mainController', function ($scope,$http,$timeout,$interval) {

    $scope.layer_display = [];
    $scope.data = {
    	'images' : [],
        'training_sessions' : [0,0,0,0,0,0,0],
        'difference_images' : [],
        'sele' : 1,
        'prev' : 2,
        'blend' : [0,11,22,33,44,55,66,77,88,100],
    };
    $scope.similar_images = [];
    $scope.label_list = [];

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
        $scope.data.prev = $scope.data.sele;
        $scope.data.sele = index;
        console.log("calling similar ...");
        $http({method:"GET" , url : "/similar/"+index , cache: false}).then(function successCallback(result) {
            console.log("... done");
            $scope.similar_images = []
            for ( var index in result.data.response ) {
              $scope.similar_images.push( { 'id' : result.data.response[index] , 'state' : 1 } );
            }
        })
    }

    $scope.learn = function(index) {
        console.log("LEARNING...");
        $http({ method : "GET" , url : "/learn/"+index , cache: false}).then(function successCallback(result) {
            console.log("...DONE");
            $scope.data.training_sessions[index] += 1;
            $scope.randomizeImage();
        })
    }

    $scope.add_label = function(label) {
        if ( $scope.label_list.indexOf(label) == -1 ) {
            $scope.label_list.push(label);
            $scope.label_list.sort();
        }
    }

    $scope.add_to_label = function(label) {
        console.log("Add these to label " + label);
        for ( var index in $scope.similar_images ) {
            var data = $scope.similar_images[index];
            if ( data.state == 1 ) {
                console.log("Found " + data.id + " as positive");
            }
        }
        console.log("Subtracting these to label " + label);
        for ( var index in $scope.similar_images ) {
            var data = $scope.similar_images[index];
            if ( data.state == 0 ) {
                console.log("Found " + data.id + " as negative");
            }
        }
        $scope.similar_images = [];

    }


});