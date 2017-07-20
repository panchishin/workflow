var imageWorkflow = angular.module('imageWorkflow' , []);


imageWorkflow.controller('mainController', function ($scope,$http,$timeout,$interval) {

    $scope.data = {
    	'images' : [],
        'training_sessions' : [0,0,0,0,0,0,0],
        'similar_images' : [],
        'difference_images' : [],
        'sele' : 1,
        'prev' : 2,
        'blend' : [0,10,20,30,40,50,60,70,80,90,100],
    }

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
            $scope.data.similar_images = result.data.response
        })
    }

    $scope.subtract = function(index) {
        console.log("calling subtract ...");
        $http({method:"GET" , url : "/difference/"+$scope.data.similar_images[0]+"/"+index , cache: false}).then(function successCallback(result) {
            console.log("... done");
            $scope.data.difference_images = result.data.response
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

});