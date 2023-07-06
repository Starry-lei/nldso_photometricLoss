//
// Created by lei on 28.04.23.
//

#include "trajectory.h"
#include <stdexcept>
#include <Eigen/LU>

Trajectory::Trajectory() {}

void Trajectory::push_back(const Mat44& pose, const Id_t id)
{
    assert_unique_id( id );

    Mat44 T_inv = pose.inverse();
	std::cout<<" \n \n show T_inv() at "<< id<< " val:\n"<< T_inv.matrix()<<std::endl;

	std::cout<<"\n show _data.empty():"<< _data.empty()<<"and !_data.empty():"<<!_data.empty()<<std::endl;

	if(!_data.empty())
	{
		std::cout << "show back() at " << id << " val:\n" << back().matrix() << std::endl;
		_data.push_back({back() * T_inv, id});
	} else
	{
		std::cout<<"\n show empty case: current id:"<< id<< " and T_inv\n "<<T_inv.matrix()<<std::endl;
		_data.push_back( {T_inv, id} );
	}




}

void Trajectory::push_back(const Mat44& pose, const Id_t id, const Id_t id2)
{
	assert_unique_id( id );
	_data.push_back({pose, id});
}

const Mat44& Trajectory::atId(const Id_t id) const
{
    auto it = find_pose_with_id(id);
    if(it == std::end(_data))
        throw std::runtime_error("could not find pose with id");

    return it->pose;
}

Mat44& Trajectory::atId(const Id_t id)
{
    auto it = find_pose_with_id(id);
    if(it == std::end(_data))
        throw std::runtime_error("could not find pose with id");

    return it->pose;
}

void Trajectory::assert_unique_id(const Id_t id) const
{
    if( std::end(_data) != find_pose_with_id(id) )
        throw std::runtime_error("duplicate id in trajectory\n");
}

EigenAlignedContainer_<Mat44> Trajectory::poses() const
{
    EigenAlignedContainer_<Mat44> ret(_data.size());
    for(size_t i = 0; i < ret.size(); ++i)
        ret[i] = _data[i].pose;

    return ret;
}

EigenAlignedContainer_<Vec3> Trajectory::cameraPositions() const
{
    EigenAlignedContainer_<Vec3> ret(_data.size());
    for(size_t i = 0; i < ret.size(); ++i)
        ret[i] = _data[i].pose.block<3,1>(0,3);

    return ret;
}


