var page = 0
var maxPage = 4
var cat = {
  x : 250,
  y : 200,
  xOffBoard1 : false,
  xOffBoard2 : false,
  yOffBoard1 : false,
  yOffBoard2 : false,



 // a : Math.pow(cat.x-mouseX,2),
 // b : Math.pow(cat.y-mouseY,2),
 // c : Math.sqrt(cat.a + cat.b),
 x2 : 150,
  y2 : 200,
}

var catBowlX=200
var catBowlChange=3
var catFoodX=[];
var catFoodY=[];
var catFoodScore = 0

function preload(){
   blackCat = loadImage("blackCat.png");
   greyCat = loadImage("greyCat.png");
}

function setup() {
        rectMode(CENTER)

  createCanvas(400, 400);
  for (var i = 0; i<50 ;i++){
    catFoodX[i] = random(mouseX-50,mouseX+50)
    catFoodY[i] = random(-400,0)
  }
}

function draw() {
  background(220);
  if (page == 0){
    //game 1
    page1()
  }
   if (page == 1){
     page2()
  }
   if (page == 2){
     page3()
  }
}

function keyPressed(){//turns page in book
  if (keyCode === RIGHT_ARROW && page<maxPage){
    page++
  }
  if (keyCode === LEFT_ARROW && page>0){
    page--
  }
}

function page1(){
    fill(0,255,0)
    rect(200,200,30,30)
}
function page2(){
      fill(0,255,0)
      image (greyCat, cat.x,cat.y,40,40)
      if (cat.xOffBoard1) {
         cat.x+=2
        if (cat.x >= 50){
          cat.xOffBoard1 = false
        } 
      }
    if (cat.xOffBoard2) {
         cat.x-=2
        if (cat.x <= 350){
          cat.xOffBoard2 = false
        } 
      }
  if (cat.yOffBoard1) {
         cat.y+=2
        if (cat.y >= 50){
          cat.yOffBoard1 = false
        } 
      }
  if (cat.yOffBoard2) {
         cat.y-=2
        if (cat.y <= 350){
          cat.yOffBoard2 = false
        } 
      }
      if (Math.sqrt(Math.pow(cat.x-mouseX,2) + Math.pow(cat.y-mouseY,2)) < 200){
        // cat.x -= 10/( mouseX - cat.x )
        // cat.y -= 10/( mouseY - cat.y )
        cat.x -=  (mouseX - cat.x )/40
        cat.y -= ( mouseY - cat.y )/40
      }
      if (cat.x > 420){
         cat.x = -19
        cat.xOffBoard1 = true
      } else if (cat.x < -20) {
         cat.x = 419
        cat.xOffBoard2 = true
      } 
  
       if (cat.y > 420) {
         cat.y = -19
        cat.yOffBoard1 = true
      } else if (cat.y < -20) {
         cat.y = 419
        cat.yOffBoard2 = true
      } 
  
       image ( blackCat, cat.x2,cat.y2,40,40)
  if (cat.x2OffBoard1) {
         cat.x2+=2
        if (cat.x2 >= 50){
          cat.x2OffBoard1 = false
        } 
      }
    if (cat.x2OffBoard2) {
         cat.x2-=2
        if (cat.x2 <= 350){
          cat.x2OffBoard2 = false
        } 
      }
  if (cat.y2OffBoard1) {
         cat.y2+=2
        if (cat.y2 >= 50){
          cat.y2OffBoard1 = false
        } 
      }
  if (cat.y2OffBoard2) {
         cat.y2-=2
        if (cat.y2 <= 350){
          cat.y2OffBoard2 = false
        } 
      }
      if (Math.sqrt(Math.pow(cat.x2-mouseX,2) + Math.pow(cat.y2-mouseY,2)) < 200){
        // cat.x -= 10/( mouseX - cat.x )
        // cat.y -= 10/( mouseY - cat.y )
        cat.x2 -=  (mouseX - cat.x2 )/40
        cat.y2 -= ( mouseY - cat.y2 )/40
      }
    
      if (cat.x2 > 420){
         cat.x2 = -19
        cat.x2OffBoard1 = true
      } else if (cat.x2 < -20) {
         cat.x2 = 419
        cat.x2OffBoard2 = true
      } 
  
       if (cat.y2 > 420) {
         cat.y2 = -19
        cat.y2OffBoard1 = true
      } else if (cat.y2 < -20) {
         cat.y2 = 419
        cat.y2OffBoard2 = true
      } 
      // cat.y%=400
      // cat.y2%=400
      // cat.x%=400
      // cat.x2%=400
      
}
function page3(){
  
  for(let i = 0; i <50; i = i +1){
      
      catFoodY[i]+=7
      fill(150,42,42)
      circle(catFoodX[i], catFoodY[i],10);
      
      if(catFoodY[i]>400){
         if(catFoodX[i]>=catBowlX-75&&catFoodX[i]<=catBowlX+75){
          catFoodScore++
          
        }
        catFoodY[i]=0;
        catFoodX[i] = random(-50,50) + mouseX
       
      }
    text('Cat food in bowl: '+ catFoodScore,20,20);
  }
  catBowlX+=catBowlChange
  if(catBowlX>=325||catBowlX<=75){
    catBowlChange*=-1
    //catBowlChange*=-(Math.random()+0.1)
  }
  rect(catBowlX,450,150)
  
}
