const WEAPON = {};

WEAPON.basic = function(pl) {
    emitBullets(pl.pos.x, pl.pos.y, -90, [0], 5, 5, BULLET.small, true);
    
    emitBullets(pl.pos.x - 20, pl.pos.y, -90, [-20,-10,0,10,20], 5, 5, BULLET.small, true);
    emitBullets(pl.pos.x + 20, pl.pos.y, -90, [-20,-10,0,10,20], 5, 5, BULLET.small, true);
};

WEAPON.dualFire = function(pl) {
    emitBullets(pl.pos.x - 20, pl.pos.y, -90, [-20,-10,0,10,20], 5, 5, BULLET.small, true);
    emitBullets(pl.pos.x + 20, pl.pos.y, -90, [-20,-10,0,10,20], 5, 5, BULLET.small, true);
};

WEAPON.tripleFire = function(pl) {
    emitBullets(pl.pos.x, pl.pos.y, -90, [-8, 0, 8], 5, 5, BULLET.small, true);
};

WEAPON.bigBomb = function(pl) {
    emitBullets(pl.pos.x, pl.pos.y, -90, [random(-10,10)], 5, 5, BULLET.large, true);
};
